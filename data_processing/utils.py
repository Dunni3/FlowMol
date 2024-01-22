import torch
import dgl

def build_edge_idxs(n_atoms: int):
    """Builds an array of edge indices for a molecule with n_atoms.
    
    The edge indicies are constructed such that the upper-triangle of the adjacency matrix is traversed before the lower triangle.
    Much of our infrastructure relies on this particular ordering of edge indicies within our graph objects.
    """
    # get upper triangle of adjacency matrix
    upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)

    # get lower triangle edges by swapping source and destination of upper_edge_idxs
    lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

    edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
    return edges

def get_upper_edge_mask(g: dgl.DGLGraph):
        """Returns a boolean mask for the edges that lie in the upper triangle of the adjacency matrix for each molecule in the batch."""
        # this algorithm assumes that the edges are ordered such that the upper triangle edges come first, followed by the lower triangle edges for each graph in the batch
        # and then those graph-wise edges are concatenated together
        # you can see that this is indeed how the edges are constructed by inspecting data_processing.dataset.MoleculeDataset.__getitem__
        edges_per_mol = g.batch_num_edges()
        ul_pattern = torch.tensor([1,0]).repeat(g.batch_size).to(g.device)
        n_edges_pattern = (edges_per_mol/2).int().repeat_interleave(2)
        upper_edge_mask = ul_pattern.repeat_interleave(n_edges_pattern).bool()
        return upper_edge_mask

def get_node_batch_idxs(g: dgl.DGLGraph):
    """Returns a tensor of integers indicating which molecule each node belongs to."""
    node_batch_idx = torch.arange(g.batch_size, device=g.device)
    node_batch_idx = node_batch_idx.repeat_interleave(g.batch_num_nodes())
    return node_batch_idx

def get_edge_batch_idxs(g: dgl.DGLGraph):
    """Returns a tensor of integers indicating which molecule each edge belongs to."""
    edge_batch_idx = torch.arange(g.batch_size, device=g.device)
    edge_batch_idx = edge_batch_idx.repeat_interleave(g.batch_num_edges())
    return edge_batch_idx

def get_batch_idxs(g: dgl.DGLGraph):
    """Returns two tensors of integers indicating which molecule each node and edge belongs to."""
    node_batch_idx = get_node_batch_idxs(g)
    edge_batch_idx = get_edge_batch_idxs(g)
    return node_batch_idx, edge_batch_idx