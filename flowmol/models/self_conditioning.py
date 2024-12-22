import torch
import torch.nn as nn
import dgl
import dgl.function as fn

from flowmol.utils.embedding import rbf_twoscale, _rbf
from flowmol.models.gvp import _norm_no_nan

class SelfConditioningResidualLayer(nn.Module):
    def __init__(self, 
                 n_atom_types,
                 n_charges,
                 n_bond_types,
                 node_embedding_dim, 
                 edge_embedding_dim,
                 rbf_dim,
                 rbf_dmax,):
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax

        self.node_residual_mlp = nn.Sequential(
            nn.Linear(node_embedding_dim+n_atom_types+n_charges+rbf_dim, node_embedding_dim),
            nn.SiLU(),
            nn.Linear(node_embedding_dim, node_embedding_dim),
            nn.SiLU(),
        )

        self.edge_residual_mlp = nn.Sequential(
            nn.Linear(edge_embedding_dim + n_bond_types + rbf_dim, edge_embedding_dim),
            nn.SiLU(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim),
            nn.SiLU(),
        )

    def forward(self, 
                g: torch.Tensor, 
                s_t: torch.Tensor,
                x_t: torch.Tensor,
                v_t: torch.Tensor,
                e_t: torch.Tensor,
                dst_dict: torch.Tensor,
                node_batch_idx: torch.Tensor,
                upper_edge_mask: torch.Tensor,):
        

        # get distances between each node in current timestep and the same node at t=1
        d_node = _norm_no_nan(x_t - dst_dict['x']) # has shape n_nodes x 1 (hopefully)
        d_node = _rbf(d_node, D_max=self.rbf_dmax, D_count=self.rbf_dim)

        node_residual_inputs = [
            s_t,
            dst_dict['a'],
            dst_dict['c'],
            d_node,
        ]
        node_residual = self.node_residual_mlp(torch.cat(node_residual_inputs, dim=-1))

        # get the edge length of every edge in g at time t and also the edge length at t=1
        d_edge_t = self.edge_distances(g, node_positions=x_t)
        d_edge_1 = self.edge_distances(g, node_positions=dst_dict['x'])

        # take only upper-triangle edges, for efficiency
        d_edge_t = d_edge_t[upper_edge_mask]
        d_edge_1 = d_edge_1[upper_edge_mask]

        edge_residual_inputs = [
            e_t[upper_edge_mask], # current state of the edge
            dst_dict['e'], # final state of the edge
            d_edge_1 - d_edge_t, # change in edge length
        ]
        edge_residual = self.edge_residual_mlp(torch.cat(edge_residual_inputs, dim=-1))

        node_feats_out = s_t + node_residual
        positions_out = x_t
        vectors_out = v_t

        edge_feats_out = torch.zeros_like(e_t)
        one_triangle_output = e_t[upper_edge_mask] + edge_residual
        edge_feats_out[upper_edge_mask] = one_triangle_output
        edge_feats_out[~upper_edge_mask] = one_triangle_output
        

        return node_feats_out, positions_out, vectors_out, edge_feats_out


    def edge_distances(self, g: dgl.DGLGraph, node_positions=None):
        """Precompute the pairwise distances between all nodes in the graph."""

        with g.local_scope():

            if node_positions is None:
                g.ndata['x_d'] = g.ndata['x_t']
            else:
                g.ndata['x_d'] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
            # x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        
        return d