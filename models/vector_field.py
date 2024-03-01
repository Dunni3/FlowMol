import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from typing import Union
import scipy

from .gvp import GVPConv, GVP, _rbf, _norm_no_nan
from .interpolant_scheduler import InterpolantScheduler
from utils.dirflow import DirichletConditionalFlow

class EndpointVectorField(nn.Module):

    def __init__(self, n_atom_types: int,
                    canonical_feat_order: list,
                    interpolant_scheduler: InterpolantScheduler,
                    n_charges: int = 6,
                    n_bond_types: int = 5, 
                    n_vec_channels: int = 16,
                    n_cp_feats: int = 0, 
                    n_hidden_scalars: int = 64,
                    n_hidden_edge_feats: int = 64,
                    n_recycles: int = 1,
                    n_molecule_updates: int = 2, 
                    convs_per_update: int = 2,
                    n_message_gvps: int = 3, 
                    n_update_gvps: int = 3,
                    separate_mol_updaters: bool = False,
                    message_norm: Union[float, str] = 100,
                    rbf_dmax = 20,
                    rbf_dim = 16,
                    exclude_charges: bool = False
    ):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm
        self.n_recycles = n_recycles
        self.separate_mol_updaters: bool = separate_mol_updaters
        self.exclude_charges = exclude_charges
        self.interpolant_scheduler = interpolant_scheduler
        self.canonical_feat_order = canonical_feat_order

        if self.exclude_charges:
            self.n_charges = 0

        self.convs_per_update = convs_per_update
        self.n_molecule_updates = n_molecule_updates

        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        assert n_vec_channels >= 3, 'n_vec_channels must be >= 3'


        self.scalar_embedding = nn.Sequential(
            nn.Linear(n_atom_types + n_charges + 1, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_scalars)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(n_bond_types, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats)
        )

        self.conv_layers = []
        for conv_idx in range(convs_per_update*n_molecule_updates):
            self.conv_layers.append(GVPConv(
                scalar_size=n_hidden_scalars,
                vector_size=n_vec_channels,
                n_cp_feats=n_cp_feats,
                edge_feat_size=n_hidden_edge_feats,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                message_norm=message_norm,
                rbf_dmax=rbf_dmax,
                rbf_dim=rbf_dim
            )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        # create molecule update layers
        self.node_position_updaters = nn.ModuleList([])
        self.edge_updaters = nn.ModuleList([])
        if self.separate_mol_updaters:
            n_updaters = n_molecule_updates
        else:
            n_updaters = 1
        for _ in range(n_updaters):
            self.node_position_updaters.append(NodePositionUpdate(n_hidden_scalars, n_vec_channels, n_gvps=3, n_cp_feats=n_cp_feats))
            self.edge_updaters.append(EdgeUpdate(n_hidden_scalars, n_hidden_edge_feats))


        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_atom_types + n_charges)
        )

        self.to_edge_logits = nn.Sequential(
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_bond_types)
        )

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                 node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, apply_softmax=False, remove_com=False):
        """Predict x_1 (trajectory destination) given x_t"""
        device = g.device

        with g.local_scope():
            # gather node and edge features for input to convolutions
            node_scalar_features = [
                g.ndata['a_t'],
                t[node_batch_idx].unsqueeze(-1)
            ]

            # if we are not excluding charges, include them in the node scalar features
            if not self.exclude_charges:
                node_scalar_features.append(g.ndata['c_t'])

            node_scalar_features = torch.cat(node_scalar_features, dim=-1)
            node_scalar_features = self.scalar_embedding(node_scalar_features)

            node_positions = g.ndata['x_t']

            num_nodes = g.num_nodes()

            # initialize the vector features for every node to be zeros
            node_vec_features = torch.zeros((num_nodes, self.n_vec_channels, 3), device=device)
            # i thought setting the first three channels to the identity matrix would be a good idea,
            # but this actually breaks rotational equivariance
            # node_vec_features[:, :3, :] = torch.eye(3, device=device).unsqueeze(0).repeat(num_nodes, 1, 1)

            edge_features = g.edata['e_t']
            edge_features = self.edge_embedding(edge_features)

            x_diff, d = self.precompute_distances(g)
            for recycle_idx in range(self.n_recycles):
                for conv_idx, conv in enumerate(self.conv_layers):

                    # perform a single convolution which updates node scalar and vector features (but not positions)
                    node_scalar_features, node_vec_features = conv(g, 
                            scalar_feats=node_scalar_features, 
                            coord_feats=node_positions,
                            vec_feats=node_vec_features,
                            edge_feats=edge_features,
                            x_diff=x_diff,
                            d=d
                    )

                    # every convs_per_update convolutions, update the node positions and edge features
                    if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:

                        if self.separate_mol_updaters:
                            updater_idx = conv_idx // self.convs_per_update
                        else:
                            updater_idx = 0

                        node_positions = self.node_position_updaters[updater_idx](node_scalar_features, node_positions, node_vec_features)
                        edge_features = self.edge_updaters[updater_idx](g, node_scalar_features, edge_features)

                        if not (conv_idx == len(self.conv_layers) - 1 and recycle_idx == self.n_recycles - 1):
                            x_diff, d = self.precompute_distances(g, node_positions)

            
            # predict final charges and atom type logits
            node_scalar_features = self.node_output_head(node_scalar_features)
            atom_type_logits = node_scalar_features[:, :self.n_atom_types]
            if not self.exclude_charges:
                atom_charge_logits = node_scalar_features[:, self.n_atom_types:]

            # predict the final edge logits
            ue_feats = edge_features[upper_edge_mask]
            le_feats = edge_features[~upper_edge_mask]
            edge_logits = self.to_edge_logits(ue_feats + le_feats)

            # project node positions back into zero-COM subspace
            if remove_com:
                g.ndata['x_1_pred'] = node_positions
                g.ndata['x_1_pred'] = g.ndata['x_1_pred'] - dgl.readout_nodes(g, feat='x_1_pred', op='mean')[node_batch_idx]
                node_positions = g.ndata['x_1_pred']

        # build a dictionary of predicted features
        dst_dict = {
            'x': node_positions,
            'a': atom_type_logits,
            'e': edge_logits
        }
        if not self.exclude_charges:
            dst_dict['c'] = atom_charge_logits

        # apply softmax to categorical features, if requested
        # at training time, we don't want to apply softmax because we use cross-entropy loss which includes softmax
        # at inference time, we want to apply softmax to get a vector which lies on the simplex
        if apply_softmax:
            for feat in dst_dict.keys():
                if feat in ['a', 'c', 'e']: # if this is a categorical feature
                    dst_dict[feat] = torch.softmax(dst_dict[feat], dim=-1) # apply softmax to this feature

        return dst_dict
    
    def precompute_distances(self, g: dgl.DGLGraph, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""

        with g.local_scope():

            if node_positions is None:
                g.ndata['x_d'] = g.ndata['x_t']
            else:
                g.ndata['x_d'] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
            x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        
        return x_diff, d
      
    def integrate(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, n_timesteps: int, visualize=False):
        """Integrate the trajectories of molecules along the vector field."""

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
    
        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # compute next step and set x_t = x_s
            g = self.step(g, s_i, t_i, alpha_t_i, alpha_t_prime_i, node_batch_idx, upper_edge_mask)

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
            n_frames = len(traj_frames['x'])
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in self.canonical_feat_order:
                    feat_traj = []
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)


            return g, reshaped_traj_frames
        
        return g
    
    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        
        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )

        # compute x_s for each feature and set x_t = x_s
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            x1_weight = alpha_t_prime_i[feat_idx]*(s_i - t_i)/(1 - alpha_t_i[feat_idx])
            xt_weight = 1 - x1_weight

            if feat == "e":
                g_data_src = g.edata

                # set the edge features so that corresponding upper and lower triangle edges have the same value
                x1 = torch.zeros_like(g.edata['e_0'])
                x1[upper_edge_mask] = dst_dict[feat]
                x1[~upper_edge_mask] = dst_dict[feat]
            else:
                g_data_src = g.ndata
                x1 = dst_dict[feat]

            g_data_src[f'{feat}_t'] = x1_weight*x1 + xt_weight*g_data_src[f'{feat}_t']

        return g


    def sample_conditional_path(self, g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        """Interpolate between the prior and true terminal state of the ligand."""
        # upper_edge_mask is not used here but it is needed for DirichletVectorField and we need to keep consistent
        # function signatures across vector field classes so that MolFM can use them interchangeably
        src_weights, dst_weights = self.interpolant_scheduler.interpolant_weights(t)

        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if feat == 'e':
                continue

            src_weight, dst_weight = src_weights[:, feat_idx][node_batch_idx].unsqueeze(-1), dst_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
            g.ndata[f'{feat}_t'] = src_weight * g.ndata[f'{feat}_0'] + dst_weight * g.ndata[f'{feat}_1_true']

        e_idx = self.canonical_feat_order.index('e')
        src_weight, dst_weight = src_weights[:, e_idx][edge_batch_idx].unsqueeze(-1), dst_weights[:, e_idx][edge_batch_idx].unsqueeze(-1)
        g.edata[f'e_t'] = src_weight * g.edata[f'e_0'] + dst_weight * g.edata[f'e_1_true']

        return g


class VectorField(EndpointVectorField):

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor, 
                 node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, apply_softmax=False, remove_com=False):
        
        dst_dict = super().forward(g, t, node_batch_idx, upper_edge_mask, apply_softmax, remove_com)
        dst_dict['x'] = dst_dict['x'] - g.ndata['x_t']
        return dst_dict
    
    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        
        # predict the destination of the trajectory given the current timepoint
        vec_field = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=False,
            remove_com=False
        )

        # compute x_s for each feature and set x_t = x_s
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if feat == "e":
                x_t = g.edata[f'e_t'][upper_edge_mask]
            else:
                x_t = g.ndata[f'{feat}_t']

            # x_s = x_t + vec_field*(s - t)
            x_s = x_t + vec_field[feat]*(s_i - t_i)

            # set x_t = x_s
            if feat == "e":
                x_t = torch.zeros_like(g.edata['e_0'])
                x_t[upper_edge_mask] = x_s
                x_t[~upper_edge_mask] = x_s
                g.edata[f'{feat}_t'] = x_t
            else:
                x_t = x_s
                g.ndata[f'{feat}_t'] = x_t

        # remove COM from x_t
        g.ndata['x_t'] = g.ndata['x_t'] - dgl.readout_nodes(g, feat='x_t', op='mean')[node_batch_idx]

        return g


class DirichletVectorField(EndpointVectorField):

    def __init__(self, *args, w_max=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_max = w_max
        self.categorical_condflows = {}
        self.categorical_condflows['a'] = DirichletConditionalFlow(K=self.n_atom_types, alpha_min=0, alpha_max=w_max, alpha_spacing=0.005)
        self.categorical_condflows['c'] = DirichletConditionalFlow(K=self.n_charges, alpha_min=0, alpha_max=w_max, alpha_spacing=0.005)
        self.categorical_condflows['e'] = DirichletConditionalFlow(K=self.n_bond_types, alpha_min=0, alpha_max=w_max, alpha_spacing=0.005)


        self.n_cat_dict = {
            'a': self.n_atom_types,
            'c': self.n_charges,
            'e': self.n_bond_types
        }

    def alpha_to_w(self, alpha_t):
        return alpha_t*self.w_max + 1

    def sample_conditional_path(self, g, t, node_batch_idx, edge_batch_idx, upper_edge_mask):
        """Interpolate between the prior and true terminal state of the ligand."""
        # TODO: this computation could be made more efficient by concatenating node features and edge features into a single tensor and then interpolate them all at once before splitting them back up
        alpha_t = self.interpolant_scheduler.alpha_t(t) # has shape (n_timepoints, n_feats)
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            # skip the bond orders, its too clunky to incorpoarte them into this loop due to upper/lower triangle edge indexing
            if feat == 'e':
                continue

            if feat == 'x':
                src_weights, dst_weights = 1 - alpha_t, alpha_t
                src_weight, dst_weight = src_weights[:, feat_idx][node_batch_idx].unsqueeze(-1), dst_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
                g.ndata[f'{feat}_t'] = src_weight * g.ndata[f'{feat}_0'] + dst_weight * g.ndata[f'{feat}_1_true']
            else: # now we are doing categorical node features (a,c)
                alpha_expanded = alpha_t[:, feat_idx][node_batch_idx].unsqueeze(-1)
                w_t = self.alpha_to_w(alpha_expanded)
                dirichlet_params = torch.ones_like(g.ndata[f'{feat}_1_true']) + w_t*g.ndata[f'{feat}_1_true']
                g.ndata[f'{feat}_t'] = torch.distributions.Dirichlet(dirichlet_params).sample()

        # sample condiitonal path for edge features
        e_idx = self.canonical_feat_order.index('e')
        alpha_expanded = alpha_t[:, e_idx][edge_batch_idx][upper_edge_mask].unsqueeze(-1)
        w_t = self.alpha_to_w(alpha_expanded)
        dirichlet_params = torch.ones_like(g.edata[f'e_1_true'][upper_edge_mask]) + w_t*(g.edata[f'e_1_true'][upper_edge_mask])
        ue_samples = torch.distributions.Dirichlet(dirichlet_params).sample()
        g.edata[f'e_t'][upper_edge_mask] = ue_samples
        g.edata[f'e_t'][~upper_edge_mask] = ue_samples

        return g
    
    def step(self, g: dgl.DGLGraph, s_i: torch.Tensor, t_i: torch.Tensor,
             alpha_t_i: torch.Tensor, alpha_t_prime_i: torch.Tensor,
             node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        
        # alpha_t_i has shape (n_feats,)
        
        # predict the destination of the trajectory given the current timepoint
        dst_dict = self(
            g, 
            t=torch.full((g.batch_size,), t_i, device=g.device),
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True
        )

        # take integration step for positions
        x1_weight = alpha_t_prime_i[0]*(s_i - t_i)/(1 - alpha_t_i[0])
        xt_weight = 1 - x1_weight
        x1 = dst_dict['x']
        g.ndata['x_t'] = x1_weight*x1 + xt_weight*g.ndata['x_t']

        # convert alpha values to w
        w_t = self.alpha_to_w(alpha_t_i)


        # take integration step for node categorical features
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat not in ['a', 'c']:
                continue
            w_feat = w_t[feat_idx]
            c_factor = self.categorical_condflows[feat].c_factor(
                g.ndata[f'{feat}_t'].cpu().numpy(),
                w_feat.item()
            )
            # c_factor has shape equal to x_t, which is (n_nodes, n_cat)
            c_factor = torch.from_numpy(c_factor).to(g.device)
            if torch.isnan(c_factor).any():
                # print(f'NAN cfactor after: xt.min(): {xt.min()}, out_probs.min(): {out_probs.min()}')
                print('NAN c_factor')
                c_factor = torch.nan_to_num(c_factor)

            # get possible endpoints as one-hot vectors
            eps = torch.eye(self.n_cat_dict[feat], device=g.device) # shape (n_cat, n_cat)

            # compute conditional vector fields for each possible endpoint
            x_t = g.ndata[f'{feat}_t'] # has shape (n_nodes, n_cat)
            cond_vec_fields = (eps[:, None, :] - x_t[None, :, :]) * c_factor.unsqueeze(0) # has shape (n_cat, n_nodes, n_cat)
            endpoint_probs = dst_dict[feat] # has shape (n_nodes, n_cat)
            endpoint_probs = endpoint_probs.transpose(0, 1).unsqueeze(-1) # has shape (n_cat, n_nodes, 1)
            marginal_vec_field = ( endpoint_probs * cond_vec_fields ).sum(dim=0)

            


        # compute x_s for each feature and set x_t = x_s
        # for feat_idx, feat in enumerate(self.canonical_feat_order):
        #     x1_weight = alpha_t_prime_i[feat_idx]*(s_i - t_i)/(1 - alpha_t_i[feat_idx])
        #     xt_weight = 1 - x1_weight

        #     if feat == "e":
        #         g_data_src = g.edata

        #         # set the edge features so that corresponding upper and lower triangle edges have the same value
        #         x1 = torch.zeros_like(g.edata['e_0'])
        #         x1[upper_edge_mask] = dst_dict[feat]
        #         x1[~upper_edge_mask] = dst_dict[feat]
        #     else:
        #         g_data_src = g.ndata
        #         x1 = dst_dict[feat]

        #     g_data_src[f'{feat}_t'] = x1_weight*x1 + xt_weight*g_data_src[f'{feat}_t']

        return g

class NodePositionUpdate(nn.Module):

    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3, n_cp_feats: int = 0):
        super().__init__()

        self.gvps = []
        for i in range(n_gvps):

            if i == n_gvps - 1:
                vectors_out = 1
                vectors_activation = nn.Identity()
            else:
                vectors_out = n_vec_channels
                vectors_activation = nn.Sigmoid()

            self.gvps.append(
                GVP(
                    dim_feats_in=n_scalars,
                    dim_feats_out=n_scalars,
                    dim_vectors_in=n_vec_channels,
                    dim_vectors_out=vectors_out,
                    n_cp_feats=n_cp_feats,
                    vectors_activation=vectors_activation
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates.squeeze(1)
    
class EdgeUpdate(nn.Module):

    def __init__(self, n_node_scalars, n_edge_feats):
        super().__init__()

        self.edge_update_fn = nn.Sequential(
            nn.Linear(n_node_scalars*2 + n_edge_feats, n_edge_feats),
            nn.SiLU(),
            nn.Linear(n_edge_feats, n_edge_feats),
            nn.SiLU(),
        )

        self.edge_norm = nn.LayerNorm(n_edge_feats)

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats):
        

        # get indicies of source and destination nodes
        src_idxs, dst_idxs = g.edges()

        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats
        ]

        edge_feats = self.edge_norm(edge_feats + self.edge_update_fn(torch.cat(mlp_inputs, dim=-1)))
        return edge_feats
