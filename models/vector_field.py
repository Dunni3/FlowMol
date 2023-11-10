import torch
import torch.nn as nn
import dgl
from typing import Union

from .gvp import GVPConv, GVP

class GVPVectorField(nn.Module):

    def __init__(self, n_atom_types: int, 
                    n_charges: int = 6, 
                    n_bond_types: int = 5, 
                    n_vec_channels: int = 16, 
                    n_hidden_scalars: int = 64,
                    n_hidden_edge_feats: int = 64,
                    n_molecule_updates: int = 2, 
                    convs_per_update: int = 2,
                    n_message_gvps: int = 3, 
                    n_update_gvps: int = 3,
                    message_norm: Union[float, str] = 100,
                    rbf_dmax = 20,
                    rbf_dim = 16
    ):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.n_bond_types = n_bond_types
        self.n_hidden_scalars = n_hidden_scalars
        self.n_hidden_edge_feats = n_hidden_edge_feats
        self.n_vec_channels = n_vec_channels
        self.message_norm = message_norm


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
                edge_feat_size=n_hidden_edge_feats,
                n_message_gvps=n_message_gvps,
                n_update_gvps=n_update_gvps,
                message_norm=message_norm,
                rbf_dmax=rbf_dmax,
                rbf_dim=rbf_dim
            )
            )
        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.node_postion_updater = NodePositionUpdate(n_hidden_scalars, n_vec_channels, n_gvps=3)
        self.edge_updater = EdgeUpdate(n_hidden_scalars, n_vec_channels, n_hidden_edge_feats)

    def forward(self, g: dgl.DGLGraph, t: torch.Tensor):
        """Returns the marginal vector field for the given graph."""
        pass

    def pred_dst(self, g: dgl.DGLGraph, t: torch.Tensor, node_batch_idx: torch.Tensor):
        """Predict x_1 (trajectory destination) given x_t"""
        device = g.device


        # gather node and edge features for input to convolutions
        node_scalar_features = [
            g.ndata['a_t'],
            g.ndata['c_t'],
            t[node_batch_idx].unsqueeze(-1)
        ]
        node_scalar_features = torch.cat(node_scalar_features, dim=-1)
        node_scalar_features = self.scalar_embedding(node_scalar_features)

        node_positions = g.ndata['x_t']

        num_nodes = g.num_nodes()
        node_vec_features = torch.zeros((num_nodes, self.n_vec_channels, 3), device=device)

        edge_features = g.edata['e_t']
        edge_features = self.edge_embedding(edge_features)

        for conv_idx, conv in enumerate(self.conv_layers):

            node_scalar_features, node_vec_features = conv(g, 
                    scalar_feats=node_scalar_features, 
                    coord_feats=node_positions,
                    vec_feats=node_vec_features,
                    edge_feats=edge_features
            )

            if conv_idx != 0 and (conv_idx + 1) % self.convs_per_update == 0:
                node_positions = self.node_postion_updater(node_scalar_features, node_positions, node_vec_features)



class NodePositionUpdate(nn.Module):

    def __init__(self, n_scalars, n_vec_channels, n_gvps: int = 3):

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
                    vectors_activation=vectors_activation
                )
            )
        self.gvps = nn.Sequential(*self.gvps)

    def forward(self, scalars: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor):
        _, vector_updates = self.gvps((scalars, vectors))
        return positions + vector_updates
    
class EdgeUpdate(nn.Module):

    def __init__(self, n_node_scalars, n_node_vecs, n_edge_feats):
        pass

    def forward(self, g: dgl.DGLGraph, node_scalars, edge_feats):
        

        # get indicies of source and destination nodes
        src_idxs, dst_idxs = g.edges()

        mlp_inputs = [
            node_scalars[src_idxs],
            node_scalars[dst_idxs],
            edge_feats
        ]
