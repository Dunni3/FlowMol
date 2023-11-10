import torch
from torch import nn, einsum
import dgl
import dgl.function as fn
from typing import List, Tuple, Union, Dict
import math

# helper functions
def exists(val):
    return val is not None

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

# the classes GVP, GVPDropout, and GVPLayerNorm are taken from lucidrains' geometric-vector-perceptron repository
# https://github.com/lucidrains/geometric-vector-perceptron/tree/main
# some adaptations have been made to these classes to make them more consistent with the original GVP paper/implementation
# specifically, using _norm_no_nan instead of torch's built in norm function, and the weight intialiation scheme for Wh and Wu

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    device = D.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class GVP(nn.Module):
    def __init__(
        self,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        hidden_vectors = None,
        feats_activation = nn.SiLU(),
        vectors_activation = nn.Sigmoid(),
        vector_gating = True,
        xavier_init = False
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in

        self.dim_vectors_out = dim_vectors_out
        dim_h = max(dim_vectors_in, dim_vectors_out) if hidden_vectors is None else hidden_vectors

        # create Wh and Wu matricies
        wh_k = 1/math.sqrt(dim_vectors_in)
        wu_k = 1/math.sqrt(dim_h)
        self.Wh = torch.zeros(dim_vectors_in, dim_h, dtype=torch.float32).uniform_(-wh_k, wh_k)
        self.Wu = torch.zeros(dim_h, dim_vectors_out, dtype=torch.float32).uniform_(-wu_k, wu_k)
        self.Wh = nn.Parameter(self.Wh)
        self.Wu = nn.Parameter(self.Wu)

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + dim_feats_in, dim_feats_out),
            feats_activation
        )

        # branching logic to use old GVP, or GVP with vector gating
        if vector_gating:
            self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out)
            if xavier_init:
                nn.init.xavier_uniform_(self.scalar_to_vector_gates.weight, gain=1)
                nn.init.constant_(self.scalar_to_vector_gates.bias, 0)
        else:
            self.scalar_to_vector_gates = None

        # self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out) if vector_gating else None

    def forward(self, data):
        feats, vectors = data
        b, n, _, v, c  = *feats.shape, *vectors.shape

        assert c == 3 and v == self.dim_vectors_in, 'vectors have wrong dimensions'
        assert n == self.dim_feats_in, 'scalar features have wrong dimensions'

        Vh = einsum('b v c, v h -> b h c', vectors, self.Wh)
        Vu = einsum('b h c, h u -> b u c', Vh, self.Wu)

        sh = _norm_no_nan(Vh)

        s = torch.cat((feats, sh), dim = 1)

        feats_out = self.to_feats_out(s)

        if exists(self.scalar_to_vector_gates):
            gating = self.scalar_to_vector_gates(feats_out)
            gating = gating.unsqueeze(dim = -1)
        else:
            gating = _norm_no_nan(Vu)

        vectors_out = self.vectors_activation(gating) * Vu

        # if torch.isnan(feats_out).any() or torch.isnan(vectors_out).any():
        #     raise ValueError("NaNs in GVP forward pass")

        return (feats_out, vectors_out)
    
class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x
    
class GVPDropout(nn.Module):
    """ Separate dropout for scalars and vectors. """
    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = _VDropout(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)


class GVPLayerNorm(nn.Module):
    """ Normal layer norm for scalars, nontrainable norm for vectors. """
    def __init__(self, feats_h_size, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, feats, vectors):

        normed_feats = self.feat_norm(feats)

        vn = _norm_no_nan(vectors, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True) + self.eps ) + self.eps
        normed_vectors = vectors / vn
        return normed_feats, normed_vectors
    


class GVPConv(nn.Module):

    """GVP graph convolution on a homogenous graph."""

    def __init__(self, scalar_size: int = 128, vector_size: int = 16,
                  scalar_activation=nn.SiLU, vector_activation=nn.Sigmoid,
                  n_message_gvps: int = 1, n_update_gvps: int = 1,
                  use_dst_feats: bool = False, rbf_dmax: float = 20, rbf_dim: int = 16,
                  edge_feat_size: int = 0, coords_range=10, message_norm: Union[float, str] = 10, dropout: float = 0.0,):
        
        super().__init__()

        # self.edge_type = edge_type
        # self.src_ntype = edge_type[0]
        # self.dst_ntype = edge_type[2]
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.scalar_activation = scalar_activation
        self.vector_activation = vector_activation
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.edge_feat_size = edge_feat_size
        self.use_dst_feats = use_dst_feats
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim
        self.dropout_rate = dropout
        self.message_norm = message_norm

        # create message passing function
        message_gvps = []
        for i in range(n_message_gvps):

            dim_vectors_in = vector_size
            dim_feats_in = scalar_size

            # on the first layer, there is an extra edge vector for the displacement vector between the two node positions
            if i == 0:
                dim_vectors_in += 1
                dim_feats_in += rbf_dim + edge_feat_size
                
            # if this is the first layer and we are using destination node features to compute messages, add them to the input dimensions
            if use_dst_feats and i == 0:
                dim_vectors_in += vector_size
                dim_feats_in += scalar_size

            message_gvps.append(
                GVP(dim_vectors_in=dim_vectors_in, 
                    dim_vectors_out=vector_size, 
                    dim_feats_in=dim_feats_in, 
                    dim_feats_out=scalar_size, 
                    feats_activation=scalar_activation(), 
                    vectors_activation=vector_activation(), 
                    vector_gating=True)
            )
        self.edge_message = nn.Sequential(*message_gvps)

        # create update function
        update_gvps = []
        for i in range(n_update_gvps):
            update_gvps.append(
                GVP(dim_vectors_in=vector_size, 
                    dim_vectors_out=vector_size, 
                    dim_feats_in=scalar_size, 
                    dim_feats_out=scalar_size, 
                    feats_activation=scalar_activation(), 
                    vectors_activation=vector_activation(), 
                    vector_gating=True)
            )
        self.node_update = nn.Sequential(*update_gvps)
        
        self.dropout = GVPDropout(self.dropout_rate)
        self.message_layer_norm = GVPLayerNorm(self.scalar_size)
        self.update_layer_norm = GVPLayerNorm(self.scalar_size)

        if self.message_norm == 'mean':
            self.agg_func = fn.mean
        else:
            self.agg_func = fn.sum

    def forward(self, g: dgl.DGLGraph, 
                scalar_feats: torch.Tensor,
                coord_feats: torch.Tensor,
                vec_feats: torch.Tensor,
                edge_feats: torch.Tensor = None,
                z: Union[float, torch.Tensor] = 1):
        # vec_feat has shape (n_nodes, n_vectors, 3)

        with g.local_scope():

            g.ndata['h'] = scalar_feats
            g.ndata['x'] = coord_feats
            g.ndata['v'] = vec_feats

            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feats is not None, "Edge features must be provided."
                g.edata["a"] = edge_feats

            # get vectors between node positions
            g.apply_edges(fn.u_sub_v("x", "x", "x_diff"))

            # normalize x_diff and compute rbf embedding of edge distance
            # dij = torch.norm(g.edges[self.edge_type].data['x_diff'], dim=-1, keepdim=True)
            dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
            g.edata['x_diff'] = g.edata['x_diff'] / dij
            g.edata['d'] = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)

            # compute messages on every edge
            g.apply_edges(self.message, etype=self.edge_type)

            # aggregate messages from every edge
            g.update_all(fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg"), etype=self.edge_type)
            g.update_all(fn.copy_e("vec_msg", "m"), self.agg_func("m", "vec_msg"), etype=self.edge_type)

            # get aggregated scalar and vector messages
            scalar_msg = g.nodes[self.dst_ntype].data["scalar_msg"] / z
            if isinstance(z, torch.Tensor):
                z = z.unsqueeze(-1)
            vec_msg = g.nodes[self.dst_ntype].data["vec_msg"] / z

            # dropout scalar and vector messages
            scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)

            # update scalar and vector features, apply layernorm
            scalar_feat = g.nodes[self.dst_ntype].data['h'] + scalar_msg
            vec_feat = g.nodes[self.dst_ntype].data['v'] + vec_msg
            scalar_feat, vec_feat = self.message_layer_norm(scalar_feat, vec_feat)

            # apply node update function, apply dropout to residuals, apply layernorm
            scalar_residual, vec_residual = self.node_update((scalar_feat, vec_feat))
            scalar_residual, vec_residual = self.dropout(scalar_residual, vec_residual)
            scalar_feat = scalar_feat + scalar_residual
            vec_feat = vec_feat + vec_residual
            scalar_feat, vec_feat = self.update_layer_norm(scalar_feat, vec_feat)

        return scalar_feat, vec_feat

    def message(self, edges):

        # concatenate x_diff and v on every edge to produce vector features
        vec_feats = [ edges.data["x_diff"].unsqueeze(1), edges.src["v"] ]
        if self.use_dst_feats:
            vec_feats.append(edges.dst["v"])
        vec_feats = torch.cat(vec_feats, dim=1)

        # create scalar features
        scalar_feats = [ edges.src['h'], edges.data['d'] ]
        if self.edge_feat_size > 0:
            scalar_feats.append(edges.data['a'])

        if self.use_dst_feats:
            scalar_feats.append(edges.dst['h'])

        scalar_feats = torch.cat(scalar_feats, dim=1)

        scalar_message, vector_message = self.edge_message((scalar_feats, vec_feats))

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}
