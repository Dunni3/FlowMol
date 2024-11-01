import torch
from torch_scatter import segment_csr
import dgl
from torch.distributions import Binomial
import torch.nn.functional as F

def threshold_purity_sampling(xt, x1, x1_probs, unmask_prob, mask_index, batch_size, batch_num_nodes, batch_idx, hc_thresh, device):

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
    node_unmask_prob[hc_mask] = ph[batch_idx[hc_mask]]
    lc_mask = (purities < hc_thresh) * masked_nodes # nodes which are currently masked and low-confidence
    node_unmask_prob[lc_mask] = pl[batch_idx[lc_mask]]

    will_unmask = torch.rand(xt.shape[0], device=device) < node_unmask_prob # sample nodes to unmask
    return will_unmask

def purity_sampling(
    g: dgl.DGLGraph,
    feat: str,
    xt,
    x1_probs,
    unmask_prob,
    mask_index,
    batch_size,
    batch_num_nodes,
    device,
    upper_edge_mask,):

    masked_nodes = xt == mask_index # mask of which nodes are currently unmasked
    purities = x1_probs.max(-1)[0] # the highest probability of any category for each node

    if feat == 'e':
        # recreate a version of purities that includes lower edges
        purities_ul = torch.zeros(g.num_edges(), device=device, dtype=purities.dtype)
        purities_ul[upper_edge_mask] = purities
        purities = purities_ul

        # duplicate masked_nodes for lower edges, mark all lower edges as unmasked
        masked_nodes_ul = torch.zeros(g.num_edges(), device=device, dtype=torch.bool)
        masked_nodes_ul[upper_edge_mask] = masked_nodes
        masked_nodes = masked_nodes_ul

    # compute the number of masked nodes per graph in the batch
    indptr = torch.zeros(batch_size+1, device=device, dtype=torch.long)
    indptr[1:] = batch_num_nodes.cumsum(0)
    masked_nodes_per_graph = segment_csr(masked_nodes.long(), indptr) # has shape (batch_size,)

    # set purities of unmasked nodes to -1
    purities[~masked_nodes] = -1

    # sample the number of nodes to unmask per graph
    n_unmask_per_graph = Binomial(total_count=masked_nodes_per_graph, probs=unmask_prob).sample()

    with g.local_scope():
        if feat == 'e':
            data_src = g.edata
            topk_func = dgl.topk_edges
        else:
            data_src = g.ndata
            topk_func = dgl.topk_nodes


        data_src['purity'] = purities.unsqueeze(-1)
        k = int(n_unmask_per_graph.max())

        if k != 0:
            _, topk_idxs_batched = topk_func(g, feat='purity', k=k, sortby=0)
            
            # topk_idxs contains indicies relative to each batch
            # but we need to convert them to batched-graph indicies
            topk_idxs_batched = topk_idxs_batched + indptr[:-1].unsqueeze(-1)

    # slice out the top k nodes for each graph
    if k != 0:
        col_indices = torch.arange(k, device=device).unsqueeze(0)
        mask = col_indices < n_unmask_per_graph.unsqueeze(1)
        
        # Apply mask to get only desired indices
        nodes_to_unmask = topk_idxs_batched[mask]
    else:
        nodes_to_unmask = torch.tensor([], device=device, dtype=torch.long)

    if feat == 'e':
        will_unmask = torch.zeros(g.num_edges(), dtype=torch.bool, device=device)
        will_unmask[nodes_to_unmask] = True
        will_unmask = will_unmask[upper_edge_mask]
    else:
        will_unmask = torch.zeros_like(xt, dtype=torch.bool)
        will_unmask[nodes_to_unmask] = True

    return will_unmask


def conflict_remasking(
    g: dgl.DGLGraph,
    feat: str,
    xt,
    x1_probs,
    mask_prob,
    mask_index,
    batch_size,
    batch_num_nodes,
    device,
    upper_edge_mask):

    masked_nodes = xt == mask_index # mask of which nodes are currently unmasked
    conflict_scores = F.cross_entropy(torch.log(x1_probs), xt, reduction='none', ignore_index=mask_index) # has shape (num_nodes,)

    if feat == 'e':
        # recreate a version of purities that includes lower edges
        conflict_scores_ul = torch.zeros(g.num_edges(), device=device, dtype=conflict_scores.dtype)
        conflict_scores_ul[upper_edge_mask] = conflict_scores
        conflict_scores_ul[~upper_edge_mask] = -1
        conflict_scores = conflict_scores_ul

        # duplicate masked_nodes for lower edges, mark all lower edges as unmasked
        masked_nodes_ul = torch.zeros(g.num_edges(), device=device, dtype=torch.bool)
        masked_nodes_ul[upper_edge_mask] = masked_nodes
        masked_nodes = masked_nodes_ul

    # compute the number of masked nodes per graph in the batch
    indptr = torch.zeros(batch_size+1, device=device, dtype=torch.long)
    indptr[1:] = batch_num_nodes.cumsum(0)
    masked_nodes_per_graph = segment_csr(masked_nodes.long(), indptr) # has shape (batch_size,)
    unmasked_nodes_per_graph = batch_num_nodes - masked_nodes_per_graph

    # set conflict scores of masked nodes to -1
    conflict_scores[masked_nodes] = -1

    # sample the number of nodes to mask per graph
    n_mask_per_graph = Binomial(total_count=unmasked_nodes_per_graph, probs=mask_prob).sample()

    with g.local_scope():
        if feat == 'e':
            data_src = g.edata
            topk_func = dgl.topk_edges
        else:
            data_src = g.ndata
            topk_func = dgl.topk_nodes


        data_src['conflict_score'] = conflict_scores.unsqueeze(-1)
        k = int(n_mask_per_graph.max())

        if k != 0:
            _, topk_idxs_batched = topk_func(g, feat='conflict_score', k=k, sortby=0)
            
            # topk_idxs contains indicies relative to each batch
            # but we need to convert them to batched-graph indicies
            topk_idxs_batched = topk_idxs_batched + indptr[:-1].unsqueeze(-1)

    # slice out the top k nodes for each graph
    if k != 0:
        col_indices = torch.arange(k, device=device).unsqueeze(0)
        mask = col_indices < n_mask_per_graph.unsqueeze(1)
        
        # Apply mask to get only desired indices
        nodes_to_mask = topk_idxs_batched[mask]
    else:
        nodes_to_mask = torch.tensor([], device=device, dtype=torch.long)

    if feat == 'e':
        will_mask = torch.zeros(g.num_edges(), dtype=torch.bool, device=device)
        will_mask[nodes_to_mask] = True
        will_mask = will_mask[upper_edge_mask]
    else:
        will_mask = torch.zeros_like(xt, dtype=torch.bool)
        will_mask[nodes_to_mask] = True

    return will_mask
