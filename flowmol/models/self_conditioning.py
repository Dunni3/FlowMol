import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class SelfConditioningResidualLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g: torch.Tensor, 
                s_t: torch.Tensor,
                x_t: torch.Tensor,
                v_t: torch.Tensor,
                e_t: torch.Tensor,
                dst_dict: torch.Tensor,
                node_batch_idx: torch.Tensor,
                upper_edge_mask: torch.Tensor,):
        pass