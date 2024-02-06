from torch.utils.data import Sampler, DistributedSampler
from data_processing.dataset import MoleculeDataset
import torch

class SameSizeMoleculeSampler(Sampler):

    def __init__(self, dataset: MoleculeDataset, batch_size: int, idxs: torch.Tensor = None, shuffle: bool = True, max_num_edges: int = 40000):
        super().__init__(dataset)
        self.dataset: MoleculeDataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_num_edges = max_num_edges

        if shuffle == False:
            raise NotImplementedError('shuffle=False is not implemented yet')
        
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
        for n_nodes in self.num_nodes.unique():
            n_nodes_idxs_map[int(n_nodes)] = self.idxs[torch.where(self.num_nodes == n_nodes)[0]]

        # compute weights on each number of nodes based on the number of examples in the dataset with that number of nodes
        n_nodes_arr = []
        n_examples_arr = []
        for n_nodes, idxs_with_n_nodes in n_nodes_idxs_map.items():
            n_nodes_arr.append(n_nodes)
            n_examples_arr.append(idxs_with_n_nodes.shape[0])
        n_nodes_arr = torch.tensor(n_nodes_arr)
        n_examples_arr = torch.tensor(n_examples_arr)
        weights = n_examples_arr / n_examples_arr.sum()


        # yield batches of indicies where all members of the batch have the same number of nodes
        for _ in range(len(self)):
            n_nodes_idx = torch.multinomial(weights, num_samples=1)
            n_nodes = n_nodes_arr[n_nodes_idx]

            # compute the batch size so that the number of edges in the batch is less than or equal to self.max_num_edges
            n_edges_per_mol = n_nodes**2 - n_nodes
            n_edges = n_edges_per_mol*self.batch_size
            if n_edges > self.max_num_edges:
                batch_size = self.max_num_edges // n_edges_per_mol
            else:
                batch_size = self.batch_size

            idxs_with_n_nodes = n_nodes_idxs_map[int(n_nodes)]
            if idxs_with_n_nodes.shape[0] <= batch_size:
                batch_idxs = idxs_with_n_nodes
            else:
                subsample_idxs = torch.randperm(len(idxs_with_n_nodes))[:batch_size]
                batch_idxs = idxs_with_n_nodes[subsample_idxs]

            yield batch_idxs
            

    def __len__(self):
        return self.idxs.shape[0] // self.batch_size


class SameSizeDistributedMoleculeSampler(DistributedSampler):

    def __init__(self, dataset: MoleculeDataset, batch_size: int, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size

    def __iter__(self):
        indicies = list(super().__iter__())
        indicies = torch.tensor(indicies)
        batch_sampler = SameSizeMoleculeSampler(self.dataset, self.batch_size, idxs=indicies, shuffle=self.shuffle)
        return iter(batch_sampler)
    
    def __len__(self):
        return self.num_samples // self.batch_size