import torch
from torch import nn, einsum
import dgl
import dgl.function as fn
from typing import List, Tuple, Union, Dict
import math
from dgl.nn.functional import edge_softmax

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
                  n_kq_gvps: int = 1,
                  attention: bool = False,
                  s_kq_dim: int = 32,
                  v_kq_dim: int = 32,
                  s_message_dim: int = None,
                  v_message_dim: int = None,
                  n_heads: int = 1,
                  n_expansion_gvps: int = 1,
                  use_dst_feats: bool = False, 
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
        self.s_kq_dim = s_kq_dim
        self.v_kq_dim = s_kq_dim
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


        # create message passing function
        message_gvps = []
        s_slope = (self.s_message_dim - scalar_size) / n_message_gvps
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
                dim_vectors_in += vector_size
                dim_feats_in += scalar_size

            # determine number of scalars output from this layer
            # if message size is smaller than scalar size, do linear interpolation on layer sizes through the gvps
            # otherwise, jump to final output size at first gvp and stay there to the end
            if self.s_message_dim < scalar_size:
                dim_feats_out = int(s_slope*i + scalar_size)
                if i == n_message_gvps - 1:
                    dim_feats_out = self.s_message_dim
            else:
                dim_feats_out = self.s_message_dim

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


        if attention:            
            # create key and query generating gvp function
            kq_gvps = []
            for i in range(n_kq_gvps):

                dim_feats_out = 2*s_kq_dim
                dim_vectors_out = 2*v_kq_dim

                if i == 0:
                    dim_feats_in = self.s_message_dim
                    dim_vectors_in = self.v_message_dim
                else:
                    dim_feats_in = 2*s_kq_dim
                    dim_vectors_in = 2*v_kq_dim

                kq_gvps.append(GVP(
                    dim_feats_in=dim_feats_in,
                    dim_feats_out=dim_feats_out,
                    dim_vectors_in=dim_vectors_in,
                    dim_vectors_out=dim_vectors_out,
                    n_cp_feats=n_cp_feats,
                    feats_activation=scalar_activation(), 
                    vectors_activation=vector_activation(), 
                    vector_gating=True)
                )
            self.kq_gvp = nn.Sequential(*kq_gvps)


            # create MLP that will compute (unnormalized) attention weights
            inp_dim = 2*s_kq_dim + v_kq_dim + edge_feat_size + rbf_dim
            self.att_mlp = nn.Sequential(
                # nn.LayerNorm(inp_dim),
                nn.Linear(inp_dim, inp_dim*2),
                nn.SiLU(),
                nn.Linear(inp_dim*2, inp_dim*2),
                nn.SiLU(),
                nn.Linear(inp_dim*2, self.n_heads*2),
                nn.LayerNorm(self.n_heads*2),
            )

            # compute number of features per attention head
            if self.s_message_dim % n_heads != 0 or self.v_message_dim % n_heads != 0:
                raise ValueError("Number of attention heads must divide the message size.")

            self.s_feats_per_head = self.s_message_dim // n_heads
            self.v_feats_per_head = self.v_message_dim // n_heads

            # self.n_heads = self.s_message_dim + self.v_message_dim

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

            if self.attention:
                s_kq, v_kq = self.kq_gvp((g.ndata['s'], g.ndata['v']))
                s_k, s_q = s_kq.chunk(2, dim=1)
                v_k, v_q = v_kq.chunk(2, dim=1)

                g.ndata['s_k'] = s_k # has shape (n_atoms, s_kq_dim)
                g.ndata['s_q'] = s_q
                g.ndata['v_k'] = v_k # has shape (n_atoms, v_kq_dim, 3)
                g.ndata['v_q'] = v_q

                # compute attention weights
                g.apply_edges(self.compute_attention_weights)

                # softmax attention weights
                att_weights = edge_softmax(g, g.edata['att_weights'])

                # split attention weights by scalar/vector features
                a_s = att_weights[:, :self.n_heads] # n_edges, n_heads
                a_v = att_weights[:, self.n_heads:] # n_edges, n_heads

                # expand attention weights to scalar and vector features
                a_s = a_s.repeat_interleave(self.s_feats_per_head, dim=1) # n_edges, s_message_dim
                a_v = a_v.repeat_interleave(self.v_feats_per_head, dim=1) # n_edges, v_message_dim

                g.edata['a_s'] = a_s # n_edges, s_message_dim
                g.edata['a_v'] = a_v.unsqueeze(-1) # n_edges, v_message_dim, 1

            # compute messages on every edge
            g.apply_edges(self.message)

            # if self.attenion, multiple messages by attention weights
            if self.attention:
                g.edata['scalar_msg'] = g.edata['scalar_msg'] * g.edata['a_s']
                g.edata['vec_msg'] = g.edata['vec_msg'] * g.edata['a_v']

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
            vec_feats.append(edges.dst["v"])
        vec_feats = torch.cat(vec_feats, dim=1)

        # create scalar features
        scalar_feats = [ edges.src['s'], edges.data['d'] ]
        if self.edge_feat_size > 0:
            scalar_feats.append(edges.data['ef'])

        if self.use_dst_feats:
            scalar_feats.append(edges.dst['s'])

        scalar_feats = torch.cat(scalar_feats, dim=1)

        scalar_message, vector_message = self.edge_message((scalar_feats, vec_feats))

        return {"scalar_msg": scalar_message, "vec_msg": vector_message}
    
    def compute_attention_weights(self, edges):

        # dot prduct all v_k and v_q along the last dimension
        v_kq = torch.einsum('ijk,ijk->ij', edges.src['v_k'], edges.dst['v_q']) / self.v_kq_dim**0.5

        att_weight_input = [
            edges.src['s_k'],
            edges.dst['s_q'],
            v_kq,
            edges.data['ef'],
            edges.data['d']
        ]

        att_weights = self.att_mlp(torch.cat(att_weight_input, dim=1))

        return {'att_weights': att_weights}
