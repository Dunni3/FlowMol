import torch
import dgl
from flowmol.models.vector_field import EndpointVectorField
from torch.nn.functional import one_hot
from torch.distributions.categorical import Categorical
from flowmol.data_processing.utils import get_edge_batch_idxs
import torch.nn.functional as F

from flowmol.utils.ctmc_utils import purity_sampling
from typing import Union, Callable

class CTMCVectorField(EndpointVectorField):

    # uses Continuous-Time Markov Chain (CTMC) to model the flow of cateogrical features (atom type, charge, bond order)
    # CTMC for flow-matching was originally proposed in https://arxiv.org/abs/2402.04997

    # we make some modifications to the original CTMC model:
    # our conditional trajectories interpolate along a progress coordiante alpha_t, which is a function of time t
    # where we set a different alpha_t for each data modality
    # we also do purity sampling in a slightly different way that in theory would be slightly less performant but is
    # computationally much more efficient when working with batched graphs

    def __init__(self, *args, 
                 stochasticity: float = 0.0, 
                 high_confidence_threshold: float = 0.0, 
                 dfm_type: str = 'campbell', 
                 cat_temperature_schedule: Union[str, Callable, float] = 0.05,
                 cat_temp_decay_max: float = 0.8,
                 cat_temp_decay_a: float = 2,
                 forward_weight_schedule: Union[str, Callable, float] = 'beta',
                 fw_beta_a: float = 0.25, fw_beta_b: float = 0.25, fw_beta_max: float = 10.0,
                 fake_atoms: bool = False,
                 **kwargs):
        super().__init__(*args, has_mask=True, **kwargs) # initialize endpoint vector field


        self.eta = stochasticity # default stochasticity parameter, 0 means no stochasticity
        self.hc_thresh = high_confidence_threshold # the threshold for for calling a prediction high-confidence, 0 means no purity sampling
        self.dfm_type = dfm_type
        self.fake_atoms = fake_atoms

        # configure temperature schedule for categorical features
        self.cat_temperature_schedule = cat_temperature_schedule
        self.cat_temp_decay_max = cat_temp_decay_max
        self.cat_temp_decay_a = cat_temp_decay_a
        self.cat_temp_func = self.build_cat_temp_schedule(
            cat_temperature_schedule=cat_temperature_schedule,
            cat_temp_decay_max=cat_temp_decay_max,
            cat_temp_decay_a=cat_temp_decay_a)
        
        # configure forward weight schedule
        self.forward_weight_schedule = forward_weight_schedule
        self.fw_beta_a = fw_beta_a
        self.fw_beta_b = fw_beta_b
        self.fw_beta_max = fw_beta_max
        self.forward_weight_func = self.build_fw_schedule(
            forward_weight_schedule=forward_weight_schedule,
            fw_beta_a=fw_beta_a,
            fw_beta_b=fw_beta_b,
            fw_beta_max=fw_beta_max)

        if self.dfm_type not in ['campbell', 'gat']:
            raise ValueError(f"Invalid dfm_type: {self.dfm_type}")

        self.mask_idxs = { # for each categorical feature, the index of the mask token
            'a': self.n_atom_types,
            'c': self.n_charges,
            'e': self.n_bond_types,
        }

    def build_cat_temp_schedule(self, cat_temperature_schedule, cat_temp_decay_max, cat_temp_decay_a):

        if cat_temperature_schedule == 'decay':
            cat_temp_func = lambda t: cat_temp_decay_max*torch.pow(1-t, cat_temp_decay_a)
        elif isinstance(cat_temperature_schedule, (float, int)):
            cat_temp_func = lambda t: cat_temperature_schedule
        elif callable(cat_temperature_schedule):
            cat_temp_func = cat_temperature_schedule
        else:
            raise ValueError(f"Invalid cat_temperature_schedule: {cat_temperature_schedule}")
        
        return cat_temp_func
    
    def build_fw_schedule(self, forward_weight_schedule, fw_beta_a, fw_beta_b, fw_beta_max):

        if forward_weight_schedule == 'beta':
            forward_weight_func = lambda t: 1 + fw_beta_max*torch.pow(t, fw_beta_a)*torch.pow(1-t, fw_beta_b)
        elif isinstance(forward_weight_schedule, (float, int)):
            forward_weight_func = lambda t: forward_weight_schedule
        elif callable(forward_weight_schedule):
            forward_weight_func = forward_weight_schedule
        else:
            raise ValueError(f"Invalid forward_weight_schedule: {forward_weight_schedule}")
        
        return forward_weight_func
        
    def sample_conditional_path(self, g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        # sample p(g_t|g_0,g_1)
        # this includes the standard probability path for positions and CTMC probability paths for categorical features
        # t has shape (batch_size,)
        _, alpha_t = self.interpolant_scheduler.interpolant_weights(t)
        batch_size = g.batch_size
        num_nodes = g.num_nodes()
        device = g.device

        # alpha_t has shape (batch_size, 4)

        # sample positions at time t
        x_idx = self.canonical_feat_order.index('x')
        dst_weight = alpha_t[:, x_idx][node_batch_idx].unsqueeze(-1)
        src_weight = 1 - dst_weight
        g.ndata['x_t'] = src_weight*g.ndata['x_0'] + dst_weight*g.ndata['x_1_true']

        # sample categorical node features
        t_node = t[node_batch_idx]
        for feat, feat_idx in zip(['a', 'c'], [1,2]):

            # all ground-truth categorical variables are set to one-hot representations without mask token by dataloader class
            # so here we convert to token indicies by argmaxing, and then one-hot encode again but with mask token

            # set x_t = x_1 to start
            xt = g.ndata[f'{feat}_1_true'].argmax(-1) # has shape (num_nodes,)
            alpha_t_feat = alpha_t[:, feat_idx][node_batch_idx] # has shape (num_nodes,)

            # set each node's feature to the mask token with probability 1 - alpha_t
            xt[ torch.rand(num_nodes, device=device) < 1 - alpha_t_feat ] = self.mask_idxs[feat]
            g.ndata[f'{feat}_t'] = one_hot(xt, num_classes=self.n_cat_feats[feat]+1)

        # sample categorical edge features
        num_edges = g.num_edges() / 2
        num_edges = int(num_edges)
        alpha_t_e = alpha_t[:, 3][edge_batch_idx][upper_edge_mask]
        et_upper = g.edata['e_1_true'][upper_edge_mask].argmax(-1)
        et_upper[ torch.rand(num_edges, device=device) < 1 - alpha_t_e ] = self.mask_idxs['e']
        
        n,d = g.edata['e_1_true'].shape
        e_t = torch.zeros((n,d+1), dtype=g.edata['e_1_true'].dtype, device=g.device)
        et_upper_onehot = one_hot(et_upper, num_classes=self.n_cat_feats['e']+1).float()
        e_t[upper_edge_mask] = et_upper_onehot
        e_t[~upper_edge_mask] = et_upper_onehot
        g.edata['e_t'] = e_t

        return g

    def integrate(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor, 
        upper_edge_mask: torch.Tensor, n_timesteps: int, 
        visualize=False, 
        dfm_type='campbell',
        stochasticity=8.0, 
        high_confidence_threshold=0.9,
        cat_temp_func=None,
        forward_weight_func=None,
        tspan=None,
        **kwargs):
        """Integrate the trajectories of molecules along the vector field."""
        
        # TODO: this overrides EndpointVectorField.integrate just because it has some extra arguments
        # we should refactor this so that we don't have to copy the entire function

        if cat_temp_func is None:
            cat_temp_func = self.cat_temp_func
        if forward_weight_func is None:
            forward_weight_func = self.forward_weight_func

        # get edge_batch_idx
        edge_batch_idx = get_edge_batch_idxs(g)

        # get the timepoint for integration
        if tspan is None:
            t = torch.linspace(0, 1, n_timesteps, device=g.device)
        else:
            t = tspan

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # set x_t = x_0
        for feat in self.canonical_feat_order:
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_t'] = data_src[f'{feat}_0']


        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
                    data_src = g.edata
                    split_sizes = g.batch_num_edges()
                else:
                    data_src = g.ndata
                    split_sizes = g.batch_num_nodes()

                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = data_src[f'{feat}_0'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [ init_frame ]
                traj_frames[f'{feat}_1_pred'] = []
    
        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_s_i = alpha_t[s_idx]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # determine if this is the last integration step
            if s_idx == t.shape[0] - 1:
                last_step = True
            else:
                last_step = False

            # compute next step and set x_t = x_s
            g = self.step(g, s_i, t_i, alpha_t_i, alpha_s_i, 
                alpha_t_prime_i, 
                node_batch_idx, 
                edge_batch_idx, 
                upper_edge_mask, 
                cat_temp_func=cat_temp_func,
                forward_weight_func=forward_weight_func,
                dfm_type=dfm_type,
                stochasticity=stochasticity, 
                high_confidence_threshold=high_confidence_threshold,
                last_step=last_step, 
                **kwargs)

            if visualize:
                for feat in self.canonical_feat_order:

                    if feat == "e":
                        g_data_src = g.edata
                    else:
                        g_data_src = g.ndata

                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    if feat == 'e':
                        split_sizes = g.batch_num_edges()
                    else:
                        split_sizes = g.batch_num_nodes()
                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)

                    ep_frame = g_data_src[f'{feat}_1_pred'].detach().cpu()
                    ep_frame = torch.split(ep_frame, split_sizes)
                    traj_frames[f'{feat}_1_pred'].append(ep_frame)

        # set x_1 = x_t
        for feat in self.canonical_feat_order:

            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata

            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:

            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in traj_frames.keys():
                    feat_traj = []
                    n_frames = len(traj_frames[feat])
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)


            return g, reshaped_traj_frames
        
        return g

    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_s_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, edge_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor,
             cat_temp_func: Callable,
             forward_weight_func: Callable, 
             dfm_type: str = 'campbell',
             stochasticity: float = 8.0,
             high_confidence_threshold: float = 0.9, 
             last_step: bool = False,
             inv_temp_func: Callable = None,):

        device = g.device

        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        if high_confidence_threshold is None:
            hc_thresh = self.hc_thresh
        else:
            hc_thresh = high_confidence_threshold

        if dfm_type is None:
            dfm_type = self.dfm_type

        if inv_temp_func is None:
            inv_temp_func = lambda t: 1.0
        
        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )
        
        dt = s_i - t_i

        # take integration step for positions
        x_1 = dst_dict['x']
        x_t = g.ndata['x_t']
        vf = self.vector_field(x_t, x_1, alpha_t_i[0], alpha_t_prime_i[0])
        g.ndata['x_t'] = x_t + dt*vf*inv_temp_func(t_i)

        # record predicted endpoint for visualization
        g.ndata['x_1_pred'] = x_1.detach().clone()

        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'x':
                continue

            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            xt = data_src[f'{feat}_t'].argmax(-1) # has shape (num_nodes,)

            if feat == 'e':
                xt = xt[upper_edge_mask]

            p_s_1 = dst_dict[feat]
            temperature = cat_temp_func(t_i)
            p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1) # log probabilities

            if dfm_type == 'campbell':


                xt, x_1_sampled = \
                self.campbell_step(p_1_given_t=p_s_1, 
                                xt=xt, 
                                stochasticity=eta, 
                                hc_thresh=hc_thresh, 
                                alpha_t=alpha_t_i[feat_idx], 
                                alpha_t_prime=alpha_t_prime_i[feat_idx],
                                dt=dt, 
                                batch_size=g.batch_size, 
                                batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(), 
                                n_classes=self.n_cat_feats[feat]+1,
                                mask_index=self.mask_idxs[feat],
                                last_step=last_step,
                                batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                                )

            elif dfm_type == 'gat':
                # record predicted endpoint for visualization
                x_1_sampled = torch.cat([p_s_1, torch.zeros_like(p_s_1[:, :1])], dim=-1)

                xt = self.gat_step(
                    p_1_given_t=p_s_1, 
                    xt=xt, 
                    alpha_t=alpha_t_i[feat_idx], 
                    alpha_t_prime=alpha_t_prime_i[feat_idx],
                    forward_weight=forward_weight_func(t_i),
                    dt=dt,
                    batch_size=g.batch_size,
                    batch_num_nodes=g.batch_num_edges()//2 if feat == 'e' else g.batch_num_nodes(),
                    n_classes=self.n_cat_feats[feat]+1,
                    mask_index=self.mask_idxs[feat],
                    batch_idx=edge_batch_idx[upper_edge_mask] if feat == 'e' else node_batch_idx,
                )
                                   
            
            # if we are doing edge features, we need to modify xt and x_1_sampled to have upper and lower edges
            if feat == 'e':
                e_t = torch.zeros_like(g.edata['e_t'])
                e_t[upper_edge_mask] = xt
                e_t[~upper_edge_mask] = xt
                xt = e_t

                e_1_sampled = torch.zeros_like(g.edata['e_t'])
                e_1_sampled[upper_edge_mask] = x_1_sampled
                e_1_sampled[~upper_edge_mask] = x_1_sampled
                x_1_sampled = e_1_sampled
            
            data_src[f'{feat}_t'] = xt
            data_src[f'{feat}_1_pred'] = x_1_sampled

        return g

        
    def campbell_step(self, p_1_given_t: torch.Tensor,
                      xt: torch.Tensor, 
                      stochasticity: float, 
                      hc_thresh: float, 
                      alpha_t: float, 
                      alpha_t_prime: float,
                      dt,
                      batch_size: int,
                      batch_num_nodes: torch.Tensor,
                      n_classes: int,
                      mask_index:int,
                      last_step: bool, 
                      batch_idx: torch.Tensor,
):
        x1 = Categorical(p_1_given_t).sample() # has shape (num_nodes,)

        unmask_prob = dt*( alpha_t_prime + stochasticity*alpha_t  ) / (1 - alpha_t)
        mask_prob = dt*stochasticity

        unmask_prob = torch.clamp(unmask_prob, min=0, max=1)
        mask_prob = torch.clamp(mask_prob, min=0, max=1)

        # sample which nodes will be unmasked
        if hc_thresh > 0:
            # select more high-confidence predictions for unmasking than low-confidence predictions
            will_unmask = purity_sampling(
                xt=xt, x1=x1, x1_probs=p_1_given_t, unmask_prob=unmask_prob,
                mask_index=mask_index, batch_size=batch_size, batch_num_nodes=batch_num_nodes,
                node_batch_idx=batch_idx, hc_thresh=hc_thresh, device=xt.device)
        else:
            # uniformly sample nodes to unmask
            will_unmask = torch.rand(xt.shape[0], device=xt.device) < unmask_prob
            will_unmask = will_unmask * (xt == mask_index) # only unmask nodes that are currently masked

        if not last_step:
            # compute which nodes will be masked
            will_mask = torch.rand(xt.shape[0], device=xt.device) < mask_prob
            will_mask = will_mask * (xt != mask_index) # only mask nodes that are currently unmasked

            # mask the nodes
            xt[will_mask] = mask_index

        # unmask the nodes
        xt[will_unmask] = x1[will_unmask]

        xt = one_hot(xt, num_classes=n_classes).float()
        x1 = one_hot(x1, num_classes=n_classes).float()
        return xt, x1
    
    def gat_step(self, 
                p_1_given_t: torch.Tensor,
                xt: torch.Tensor, 
                alpha_t: float, 
                alpha_t_prime: float,
                forward_weight: float,
                dt,
                batch_size: int,
                batch_num_nodes: torch.Tensor,
                n_classes: int,
                mask_index:int,
                batch_idx: torch.Tensor,
):


        # add a zero-column on to p_1_given_t to represent the mask token
        p_1_given_t = torch.cat([p_1_given_t, torch.zeros_like(p_1_given_t[:, :1])], dim=-1)

        # create a one-hot encoding of xt
        delta_xt = one_hot(xt, num_classes=n_classes).float()

        # compute forward probability velocity
        u_forward = alpha_t_prime / (1 - alpha_t) * (p_1_given_t - delta_xt)

        # create a delta on the mask token
        delta_mask = torch.zeros_like(delta_xt)
        delta_mask[:, mask_index] = 1

        # compute the backward probability velocity
        u_backward = alpha_t_prime / (alpha_t + 1e-8) * (delta_xt - delta_mask)
    
        # compute the probability velocity
        backward_weight = forward_weight - 1
        pvel = forward_weight*u_forward - backward_weight*u_backward

        # compute the parameters of the transition distritibution
        p_step = delta_xt + dt*pvel

        # clamp p_step to be valid
        p_step = torch.clamp(p_step, min=1.0e-9, max=1)

        # sample x_{t+dt} from the transition distribution
        x_dt = Categorical(p_step).sample()

        # one-hot encode x_{t+dt}
        x_dt = one_hot(x_dt, num_classes=n_classes).float()

        return x_dt
