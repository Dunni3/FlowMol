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

        # for each unique number of nodes, get the dataset indicies of all graphs in self.idxs that have that number of nodes
        n_nodes_idxs_map = {}
        n_nodes_unique = self.num_nodes.unique()
        for n_nodes in n_nodes_unique:
            n_nodes_idxs_map[int(n_nodes)] = self.idxs[torch.where(self.num_nodes == n_nodes)[0]]


        # yield batches of indicies where all members of the batch have the same number of nodes
        for _ in range(len(self)):
            n_nodes_idx = torch.randint(low=0, high=len(n_nodes_unique), size=(1,))
            n_nodes = n_nodes_unique[n_nodes_idx]
            idxs_with_n_nodes = n_nodes_idxs_map[int(n_nodes)]
            if idxs_with_n_nodes.shape[0] <= self.batch_size:
                batch_idxs = idxs_with_n_nodes
            else:
                subsample_idxs = torch.randperm(len(idxs_with_n_nodes))[:self.batch_size]
                batch_idxs = idxs_with_n_nodes[subsample_idxs]
            yield batch_idxs
            

    def __len__(self):
        return self.idxs.shape[0] // self.batch_size