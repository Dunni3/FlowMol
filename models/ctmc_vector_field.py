import torch
import dgl
from models.vector_field import EndpointVectorField
from torch.nn.functional import one_hot
from torch.distributions.categorical import Categorical
from data_processing.utils import get_edge_batch_idxs
import torch.nn.functional as F

from utils.ctmc_utils import purity_sampling

class CTMCVectorField(EndpointVectorField):

    # uses Continuous-Time Markov Chain (CTMC) to model the flow of cateogrical features (atom type, charge, bond order)
    # CTMC for flow-matching was originally proposed in https://arxiv.org/abs/2402.04997

    # we make some modifications to the original CTMC model:
    # our conditional trajectories interpolate along a progress coordiante alpha_t, which is a function of time t
    # where we set a different alpha_t for each data modality
    # we also do purity sampling in a slightly different way that in theory would be slightly less performant but is
    # computationally much more efficient when working with batched graphs

    def __init__(self, *args, stochasticity: float = 0.0, high_confidence_threshold: float = 0.0, **kwargs):
        super().__init__(*args, has_mask=True, **kwargs) # initialize endpoint vector field


        self.eta = stochasticity # default stochasticity parameter, 0 means no stochasticity
        self.hc_thresh = high_confidence_threshold # the threshold for for calling a prediction high-confidence, 0 means no purity sampling

        self.mask_idxs = { # for each categorical feature, the index of the mask token
            'a': self.n_atom_types,
            'c': self.n_charges,
            'e': self.n_bond_types,
        }

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
        visualize=False, stochasticity=None, 
        high_confidence_threshold=None,
        **kwargs):
        """Integrate the trajectories of molecules along the vector field."""
        
        # TODO: this overrides EndpointVectorField.integrate just because it has some extra arguments
        # we should refactor this so that we don't have to copy the entire function

        # get edge_batch_idx
        edge_batch_idx = get_edge_batch_idxs(g)

        # get the timepoint for integration
        t = torch.linspace(0, 1, n_timesteps, device=g.device)

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
            alpha_t_prime_i, node_batch_idx, edge_batch_idx, upper_edge_mask, 
                stochasticity=stochasticity, 
                last_step=last_step, 
                high_confidence_threshold=high_confidence_threshold,
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
             stochasticity: float = None,
             high_confidence_threshold: float = None, 
             last_step: bool = False,
             temperature: float = 1.0):

        device = g.device

        if stochasticity is None:
            eta = self.eta
        else:
            eta = stochasticity

        if high_confidence_threshold is None:
            hc_thresh = self.hc_thresh
        else:
            hc_thresh = high_confidence_threshold
        
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
        x1_weight = alpha_t_prime_i[0]*(s_i - t_i)/(1 - alpha_t_i[0])
        xt_weight = 1 - x1_weight
        x1 = dst_dict['x']
        g.ndata['x_t'] = x1_weight*x1 + xt_weight*g.ndata['x_t']

        # record predicted endpoint for visualization
        g.ndata['x_1_pred'] = x1.detach().clone()

        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat not in ['a', 'c']:
                continue
            xt = g.ndata[f'{feat}_t'].argmax(-1) # has shape (num_nodes,)

            p_s_1 = dst_dict[feat]
            p_s_1 = F.softmax(torch.log(p_s_1)/temperature, dim=-1) # log probabilities

            x1 = Categorical(p_s_1).sample() # has shape (num_nodes,)

            # record predicted endpoint for visualization
            g.ndata[f'{feat}_1_pred'] = one_hot(x1.detach().clone(), num_classes=self.n_cat_feats[feat]+1).float()

            unmask_prob = dt*( alpha_t_prime_i[feat_idx] + eta*alpha_t_i[feat_idx]  ) / (1 - alpha_t_i[feat_idx])

            # sample which nodes will be unmasked
            if hc_thresh > 0:
                # select more high-confidence predictions for unmasking than low-confidence predictions
                will_unmask = purity_sampling(
                    xt=xt, x1=x1, x1_probs=p_s_1, unmask_prob=unmask_prob,
                    mask_index=self.mask_idxs[feat], batch_size=g.batch_size, batch_num_nodes=g.batch_num_nodes(),
                    node_batch_idx=node_batch_idx, hc_thresh=hc_thresh, device=g.device)
            else:
                # uniformly sample nodes to unmask
                will_unmask = torch.rand(xt.shape[0], device=device) < unmask_prob
                will_unmask = will_unmask * (xt == self.mask_idxs[feat]) # only unmask nodes that are currently masked

            if not last_step:
                # compute which nodes will be masked
                will_mask = torch.rand(xt.shape[0], device=device) < dt*eta
                will_mask = will_mask * (xt != self.mask_idxs[feat]) # only mask nodes that are currently unmasked

                # mask the nodes
                xt[will_mask] = self.mask_idxs[feat]

            # unmask the nodes
            xt[will_unmask] = x1[will_unmask]

            g.ndata[f'{feat}_t'] = one_hot(xt, num_classes=self.n_cat_feats[feat]+1).float()

        # take integration step for bond order (e)
        feat = 'e'
        feat_idx = 3
        n_edges = g.num_edges() // 2
        xt = g.edata['e_t'][upper_edge_mask].argmax(-1) # has shape (n_edges,)
        p_e_1 = dst_dict[feat]
        p_e_1 = F.softmax(torch.log(p_e_1)/temperature, dim=-1)
        x1 = Categorical(p_e_1).sample() # has shape (n_edges,)

        # record predicted endpoint for visualization
        e_1_pred = torch.zeros((g.num_edges(), self.n_cat_feats['e']+1), device=xt.device)
        e_1_pred[upper_edge_mask] = one_hot(x1.detach().clone(), num_classes=self.n_cat_feats['e']+1).float()
        e_1_pred[~upper_edge_mask] = e_1_pred[upper_edge_mask]
        g.edata['e_1_pred'] = e_1_pred

        unmask_prob = dt*( alpha_t_prime_i[feat_idx] + eta*alpha_t_i[feat_idx]  ) / (1 - alpha_t_i[feat_idx])

        # sample which edges will be unmasked
        if hc_thresh > 0:
            will_unmask = purity_sampling(
                xt=xt, x1=x1, x1_probs=p_e_1, unmask_prob=unmask_prob,
                mask_index=self.mask_idxs[feat], batch_size=g.batch_size, batch_num_nodes=g.batch_num_edges() / 2,
                node_batch_idx=edge_batch_idx[upper_edge_mask], hc_thresh=hc_thresh, device=g.device)
        else:
            will_unmask = torch.rand(n_edges, device=device) < unmask_prob
            will_unmask = will_unmask * (xt == self.mask_idxs[feat])

        if not last_step:
            # compute which edges will be masked
            will_mask = torch.rand(n_edges, device=device) < dt*eta
            will_mask = will_mask * (xt != self.mask_idxs[feat])

            # mask the edges
            xt[will_mask] = self.mask_idxs[feat]

        # unmask the edges
        xt[will_unmask] = x1[will_unmask]

        # set the new edge features in the graph object
        e_t = torch.zeros_like(g.edata['e_t'])
        et_upper = one_hot(xt, num_classes=self.n_cat_feats['e']+1).float()
        e_t[upper_edge_mask] = et_upper
        e_t[~upper_edge_mask] = et_upper
        g.edata['e_t'] = e_t

        return g

        


