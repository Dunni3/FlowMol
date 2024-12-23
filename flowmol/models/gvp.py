import torch
from torch import nn, einsum
import dgl
import dgl.function as fn
from typing import List, Tuple, Union, Dict
import math
from dgl.nn.functional import edge_softmax
from flowmol.utils.embedding import _rbf

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



class GVP(nn.Module):
    def __init__(
        self,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        n_cp_feats = 0, # number of cross-product features added to hidden vector features
        hidden_vectors = None,
        feats_activation = nn.SiLU(),
        vectors_activation = nn.Sigmoid(),
        vector_gating = True,
        xavier_init = False
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in
        self.n_cp_feats = n_cp_feats

        self.dim_vectors_out = dim_vectors_out
        dim_h = max(dim_vectors_in, dim_vectors_out) if hidden_vectors is None else hidden_vectors

        # create Wh matrix
        wh_k = 1/math.sqrt(dim_vectors_in)
        self.Wh = torch.zeros(dim_vectors_in, dim_h, dtype=torch.float32).uniform_(-wh_k, wh_k)
        self.Wh = nn.Parameter(self.Wh)

        # create Wcp matrix if we are using cross-product features
        if n_cp_feats > 0:
            wcp_k = 1/math.sqrt(dim_vectors_in)
            self.Wcp = torch.zeros(dim_vectors_in, n_cp_feats*2, dtype=torch.float32).uniform_(-wcp_k, wcp_k)
            self.Wcp = nn.Parameter(self.Wcp)

        # create Wu matrix
        if n_cp_feats > 0: # the number of vector features going into Wu is increased by n_cp_feats if we are using cross-product features
            wu_in_dim = dim_h + n_cp_feats
        else:
            wu_in_dim = dim_h
        wu_k = 1/math.sqrt(wu_in_dim)
        self.Wu = torch.zeros(wu_in_dim, dim_vectors_out, dtype=torch.float32).uniform_(-wu_k, wu_k)
        self.Wu = nn.Parameter(self.Wu)

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + n_cp_feats + dim_feats_in, dim_feats_out),
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

        # feats has shape (batch_size, n_feats)
        # vectors has shape (batch_size, n_vectors, 3)

        assert c == 3 and v == self.dim_vectors_in, 'vectors have wrong dimensions'
        assert n == self.dim_feats_in, 'scalar features have wrong dimensions'

        Vh = einsum('b v c, v h -> b h c', vectors, self.Wh) # has shape (batch_size, dim_h, 3)
        
        # if we are including cross-product features, compute them here
        if self.n_cp_feats > 0:
            # convert dim_vectors_in vectors to n_cp_feats*2 vectors
            Vcp = einsum('b v c, v p -> b p c', vectors, self.Wcp) # has shape (batch_size, n_cp_feats*2, 3)
            # split the n_cp_feats*2 vectors into two sets of n_cp_feats vectors
            cp_src, cp_dst = torch.split(Vcp, self.n_cp_feats, dim=1) # each has shape (batch_size, n_cp_feats, 3)
            # take the cross product of the two sets of vectors
            cp = torch.linalg.cross(cp_src, cp_dst, dim=-1) # has shape (batch_size, n_cp_feats, 3)

            # add the cross product features to the hidden vector features
            Vh = torch.cat((Vh, cp), dim=1) # has shape (batch_size, dim_h + n_cp_feats, 3)

        Vu = einsum('b h c, h u -> b u c', Vh, self.Wu) # has shape (batch_size, dim_vectors_out, 3)

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

    def forward(self, data):
        feats, vectors = data

        normed_feats = self.feat_norm(feats)

        vn = _norm_no_nan(vectors, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True) + self.eps ) + self.eps
        normed_vectors = vectors / vn
        return normed_feats, normed_vectors
    


class GVPConv(nn.Module):

    """GVP graph convolution on a homogenous graph."""

    def __init__(self, 
                  scalar_size: int = 128, 
                  vector_size: int = 16, 
                  n_cp_feats: int = 0,
                  scalar_activation=nn.SiLU, 
                  vector_activation=nn.Sigmoid,
                  n_message_gvps: int = 1, 
                  n_update_gvps: int = 1,
                  attention: bool = False,
                  s_message_dim: int = None,
                  v_message_dim: int = None,
                  n_heads: int = 1,
                  n_expansion_gvps: int = 1,
                  use_dst_feats: bool = False, 
                  dst_feat_msg_reduction_factor: float = 4,
                  rbf_dmax: float = 20, 
                  rbf_dim: int = 16,
                  edge_feat_size: int = 0, 
                  message_norm: Union[float, str] = 10, 
                  dropout: float = 0.0,
                  ):
        
        super().__init__()

        # self.edge_type = edge_type
        # self.src_ntype = edge_type[0]
        # self.dst_ntype = edge_type[2]
        self.scalar_size = scalar_size
        self.vector_size = vector_size
        self.n_cp_feats = n_cp_feats
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

        # dims for message reduction and also attention
        self.s_message_dim = s_message_dim
        self.v_message_dim = v_message_dim
        self.attention = attention
        self.n_heads = n_heads

        if s_message_dim is None:
            self.s_message_dim = scalar_size
        
        if v_message_dim is None:
            self.v_message_dim = vector_size

        # determine whether we are performing compressed message passing
        if self.s_message_dim != scalar_size or self.v_message_dim != vector_size:
            self.compressed_messaging = True
        else:
            self.compressed_messaging = False


        # if message size is smaller than node embedding size, we need to project node features down to message size
        if self.compressed_messaging:
            compression_gvps = []
            for i in range(n_expansion_gvps): # implicit here that n_expansion_gvps is the same as n_compression_gvps
                if i == 0:
                    dim_feats_in = scalar_size
                    dim_vectors_in = vector_size
                else:
                    dim_feats_in = max(self.s_message_dim, scalar_size)
                    dim_vectors_in = max(self.v_message_dim, vector_size)

                if i == n_expansion_gvps - 1:
                    dim_feats_out = self.s_message_dim
                    dim_vectors_out = self.v_message_dim
                else:
                    dim_feats_out = max(self.s_message_dim, scalar_size)
                    dim_vectors_out = max(self.v_message_dim, vector_size)

                compression_gvps.append(
                    GVP(dim_vectors_in=dim_vectors_in, 
                        dim_vectors_out=dim_vectors_out,
                        n_cp_feats=n_cp_feats,
                        dim_feats_in=dim_feats_in, 
                        dim_feats_out=dim_feats_out,
                        feats_activation=scalar_activation(), 
                        vectors_activation=vector_activation(), 
                        vector_gating=True)
                )
            self.node_compression = nn.Sequential(*compression_gvps)
        else:
            self.node_compression = nn.Identity()

        if attention:            
            # compute number of features per attention head
            if self.s_message_dim % n_heads != 0 or self.v_message_dim % n_heads != 0:
                raise ValueError("Number of attention heads must divide the message size.")

            self.s_feats_per_head = self.s_message_dim // n_heads
            self.v_feats_per_head = self.v_message_dim // n_heads
            extra_scalar_feats = n_heads*2

            self.att_weight_projection = nn.Linear(extra_scalar_feats, extra_scalar_feats, bias=False)

        else:
            extra_scalar_feats = 0

        if use_dst_feats:
            if dst_feat_msg_reduction_factor != 1:
                s_dst_feats_for_messages = int(self.s_message_dim/dst_feat_msg_reduction_factor)
                v_dst_feats_for_messages = int(self.v_message_dim/dst_feat_msg_reduction_factor)
                self.dst_feat_msg_projection = GVP(
                    dim_vectors_in=v_message_dim,
                    dim_vectors_out=v_dst_feats_for_messages,
                    dim_feats_in=s_message_dim,
                    dim_feats_out=s_dst_feats_for_messages,
                    n_cp_feats=0,
                    feats_activation=scalar_activation(),
                    vectors_activation=vector_activation(),
                )
            else:
                s_dst_feats_for_messages = self.s_message_dim
                v_dst_feats_for_messages = self.v_message_dim
                self.dst_feat_msg_projection = nn.Identity()
        else:
            s_dst_feats_for_messages = 0
            v_dst_feats_for_messages = 0


        # create message passing function
        message_gvps = []
        s_slope = (self.s_message_dim + extra_scalar_feats - scalar_size) / n_message_gvps
        v_slope = (self.v_message_dim - vector_size) / n_message_gvps
        for i in range(n_message_gvps):

            dim_vectors_in = self.v_message_dim
            dim_feats_in = self.s_message_dim

            # on the first layer, there is an extra edge vector for the displacement vector between the two node positions
            if i == 0:
                dim_vectors_in += 1 
                dim_feats_in += rbf_dim + edge_feat_size
            else:
                # if not first layer, input size is the output size of the previous layer
                dim_feats_in = dim_feats_out
                dim_vectors_in = dim_vectors_out
                
            # if this is the first layer and we are using destination node features to compute messages, add them to the input dimensions
            if use_dst_feats and i == 0:
                dim_vectors_in += v_dst_feats_for_messages
                dim_feats_in += s_dst_feats_for_messages

            # determine number of scalars output from this layer
            # if message size is smaller than scalar size, do linear interpolation on layer sizes through the gvps
            # otherwise, jump to final output size at first gvp and stay there to the end
            if self.s_message_dim < scalar_size:
                dim_feats_out = int(s_slope*i + scalar_size)
                if i == n_message_gvps - 1:
                    dim_feats_out = self.s_message_dim + extra_scalar_feats
            else:
                dim_feats_out = self.s_message_dim + extra_scalar_feats

            # same logic applied to the number of vectors output from this layer
            if self.v_message_dim < vector_size:
                dim_vectors_out = int(v_slope*i + vector_size)
                if i == n_message_gvps - 1:
                    dim_vectors_out = self.v_message_dim
            else:
                dim_vectors_out = self.v_message_dim


            message_gvps.append(
                GVP(dim_vectors_in=dim_vectors_in, 
                    dim_vectors_out=dim_vectors_out,
                    n_cp_feats=n_cp_feats,
                    dim_feats_in=dim_feats_in, 
                    dim_feats_out=dim_feats_out,
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
                    n_cp_feats=n_cp_feats,
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

        if isinstance(self.message_norm, str) and self.message_norm not in ['mean', 'sum']:
            raise ValueError(f"message_norm must be either 'mean', 'sum', or a number, got {self.message_norm}")
        else:
            if self.message_norm not in ['mean', 'sum']:
                assert isinstance(self.message_norm, (float, int)), "message_norm must be either 'mean', 'sum', or a number"

        if self.message_norm == 'mean':
            self.agg_func = fn.mean
        else:
            self.agg_func = fn.sum


        # if message size is smaller than node embedding size, we need to project aggregated messages back to the node embedding size
        if self.compressed_messaging:
            projection_gvps = []
            for i in range(n_expansion_gvps):
                if i == 0:
                    dim_feats_in = self.s_message_dim
                    dim_vectors_in = self.v_message_dim
                else:
                    dim_feats_in = scalar_size
                    dim_vectors_in = vector_size

                dim_feats_out = scalar_size
                dim_vectors_out = vector_size

                projection_gvps.append(
                    GVP(dim_vectors_in=dim_vectors_in, 
                        dim_vectors_out=dim_vectors_out,
                        n_cp_feats=n_cp_feats,
                        dim_feats_in=dim_feats_in, 
                        dim_feats_out=dim_feats_out,
                        feats_activation=scalar_activation(), 
                        vectors_activation=vector_activation(), 
                        vector_gating=True)
                )
            self.message_expansion = nn.Sequential(*projection_gvps)
        else:
            self.message_expansion = nn.Identity()

    def forward(self, g: dgl.DGLGraph, 
                scalar_feats: torch.Tensor,
                coord_feats: torch.Tensor,
                vec_feats: torch.Tensor,
                edge_feats: torch.Tensor = None,
                x_diff: torch.Tensor = None,
                d: torch.Tensor = None):
        # vec_feat has shape (n_nodes, n_vectors, 3)

        with g.local_scope():

            g.ndata['s'] = scalar_feats
            g.ndata['x'] = coord_feats
            g.ndata['v'] = vec_feats

            if x_diff is not None and d is not None:
                g.edata['x_diff'] = x_diff
                g.edata['d'] = d

            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feats is not None, "Edge features must be provided."
                g.edata["ef"] = edge_feats

            # normalize x_diff and compute rbf embedding of edge distance
            # dij = torch.norm(g.edges[self.edge_type].data['x_diff'], dim=-1, keepdim=True)
            if 'x_diff' not in g.edata:
                # get vectors between node positions
                g.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
                dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
                g.edata['x_diff'] = g.edata['x_diff'] / dij
                g.edata['d'] = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)


            # apply node compression
            g.ndata['s'], g.ndata['v'] = self.node_compression((g.ndata['s'], g.ndata['v']))

            if self.use_dst_feats:
                g.ndata['s_dst_msg'] , g.ndata['v_dst_msg'] = self.dst_feat_msg_projection((g.ndata['s'], g.ndata['v']))

            # compute messages on every edge
            g.apply_edges(self.message)

            # if self.attenion, multiple messages by attention weights
            if self.attention:
                scalar_msg, att_logits = g.edata['scalar_msg'][:, :self.s_message_dim], g.edata['scalar_msg'][:, self.s_message_dim:]
                att_logits = self.att_weight_projection(att_logits)
                att_weights = edge_softmax(g, att_logits)
                s_att_weights = att_weights[:, :self.n_heads]
                v_att_weights = att_weights[:, self.n_heads:]
                s_att_weights = s_att_weights.repeat_interleave(self.s_feats_per_head, dim=1)
                v_att_weights = v_att_weights.repeat_interleave(self.v_feats_per_head, dim=1)
                g.edata['scalar_msg'] = scalar_msg * s_att_weights
                g.edata['vec_msg'] = g.edata['vec_msg'] * v_att_weights.unsqueeze(-1)

            # aggregate messages from every edge
            g.update_all(fn.copy_e("scalar_msg", "m"), self.agg_func("m", "scalar_msg"))
            g.update_all(fn.copy_e("vec_msg", "m"), self.agg_func("m", "vec_msg"))

            # get aggregated scalar and vector messages
            if isinstance(self.message_norm, str):
                z = 1
            else:
                z = self.message_norm

            scalar_msg = g.ndata["scalar_msg"] / z
            vec_msg = g.ndata["vec_msg"] / z

            # apply projection (expansion) to aggregated messages
            scalar_msg, vec_msg = self.message_expansion((scalar_msg, vec_msg))

            # dropout scalar and vector messages
            scalar_msg, vec_msg = self.dropout(scalar_msg, vec_msg)

            # update scalar and vector features, apply layernorm
            scalar_feat = scalar_feats + scalar_msg
            vec_feat = vec_feats + vec_msg
            scalar_feat, vec_feat = self.message_layer_norm((scalar_feat, vec_feat))

            # apply node update function, apply dropout to residuals, apply layernorm
            scalar_residual, vec_residual = self.node_update((scalar_feat, vec_feat))
            scalar_residual, vec_residual = self.dropout(scalar_residual, vec_residual)
            scalar_feat = scalar_feat + scalar_residual
            vec_feat = vec_feat + vec_residual
            scalar_feat, vec_feat = self.update_layer_norm((scalar_feat, vec_feat))

        return scalar_feat, vec_feat

    def message(self, edges):

        # concatenate x_diff and v on every edge to produce vector features
        vec_feats = [ edges.data["x_diff"].unsqueeze(1), edges.src["v"] ]
        if self.use_dst_feats:
            vec_feats.append(edges.dst["v_dst_msg"])
        vec_feats = torch.cat(vec_feats, dim=1)

        # create scalar features
        scalar_feats = [ edges.src['s'], edges.data['d'] ]
        if self.edge_feat_size > 0:
            scalar_feats.append(edges.data['ef'])

        if self.use_dst_feats:
            scalar_feats.append(edges.dst['s_dst_msg'])

        scalar_feats = torch.cat(scalar_feats, dim=1)

        scalar_message, vector_message = self.edge_message((scalar_feats, vec_feats))

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}
