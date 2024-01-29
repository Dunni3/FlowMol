from torch.utils.data import Sampler, DistributedSampler
from data_processing.dataset import MoleculeDataset
import torch

class SameSizeMoleculeSampler(Sampler):

    def __init__(self, dataset: MoleculeDataset, batch_size, idxs: torch.Tensor = None, shuffle: bool = True):
        super().__init__(dataset)
        self.dataset: MoleculeDataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if idxs is None:
            self.idxs = torch.arange(len(dataset))
        else:
            self.idxs = idxs


        node_idx_array = self.dataset.node_idx_array

        if idxs is not None:
            node_idx_array = node_idx_array[idxs]

        self.num_nodes = node_idx_array[:, 1] - node_idx_array[:, 0] # array of shape (indicies.shape[0],) containing the number of nodes in each graph

    def __iter__(self):

        n_nodes_idxs_map = {}
        for n_nodes in self.num_nodes.unique():
            n_nodes_idxs_map[int(n_nodes)] = self.idxs[torch.where(self.num_nodes == n_nodes)[0]]