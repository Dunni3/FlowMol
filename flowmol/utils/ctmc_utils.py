import torch
from torch_scatter import segment_csr

def purity_sampling(xt, x1, x1_probs, unmask_prob, mask_index, batch_size, batch_num_nodes, node_batch_idx, hc_thresh, device):

    masked_nodes = xt == mask_index # mask of which nodes are currently unmasked
    purities = x1_probs.max(-1)[0] # the highest probability of any category for each node

    hc_mask = purities >= hc_thresh # mask of which nodes are high-confidence
    hc_mask = hc_mask * masked_nodes # only consider nodes that are currently masked

    # compute the number of hc nodes in each graph in the batch
    indptr = torch.zeros(batch_size+1, device=device, dtype=torch.long)
    indptr[1:] = batch_num_nodes.cumsum(0)
    hc_nodes_per_graph = segment_csr(hc_mask.long(), indptr) # has shape (batch_size,)

    # compute the number of masked nodes in each graph in the batch
    masked_nodes_per_graph = segment_csr(masked_nodes.long(), indptr) # has shape (batch_size,)

    # compute max value of ph for each graph in the batch
    ph_max = unmask_prob*masked_nodes_per_graph / hc_nodes_per_graph
    ph_max[ hc_nodes_per_graph == 0 ] = torch.inf

    # compute ph and pl for each graph in the batch
    ph = torch.minimum(ph_max, torch.full_like(ph_max, 1.0)) # bernoulli trial probability of high confidence nodes in each graph
    pl = (unmask_prob*masked_nodes_per_graph - ph*hc_nodes_per_graph) / (masked_nodes_per_graph - hc_nodes_per_graph) # bernoulli trial probability of low confidence nodes in each graph

    # construct a tensor containing the unmask probability for every node
    node_unmask_prob = torch.zeros_like(xt).float()
    node_unmask_prob[hc_mask] = ph[node_batch_idx[hc_mask]]
    lc_mask = (purities < hc_thresh) * masked_nodes # nodes which are currently masked and low-confidence
    node_unmask_prob[lc_mask] = pl[node_batch_idx[lc_mask]]

    will_unmask = torch.rand(xt.shape[0], device=device) < node_unmask_prob # sample nodes to unmask
    return will_unmask