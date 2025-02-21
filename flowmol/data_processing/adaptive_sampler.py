import torch
from torch.utils.data import Sampler, BatchSampler

from flowmol.data_processing.dataset import MoleculeDataset

class AdaptiveEdgeSampler(BatchSampler):
    def __init__(self, dataset, edges_per_batch: int,
                 sampler = None,
                 distributed: bool = False,
                 rank: int = None,
                 num_replicas: int = None,
                  ):

        self.dataset: MoleculeDataset = dataset
        self.edges_per_batch = edges_per_batch
        self.distributed = distributed

        if self.distributed:
            self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
            self.rank = rank if rank is not None else torch.distributed.get_rank()

            dataset_frac_per_worker = 1.0 / self.num_replicas
            self.frac_start = self.rank * dataset_frac_per_worker
            self.frac_end = (self.rank + 1) * dataset_frac_per_worker
        else:
            self.rank = 0
            self.num_replicas = 1
            self.frac_start = 0
            self.frac_end = 1

        self.samples_per_epoch = len(self.dataset) // self.num_replicas
        # edges_per_sample = (44*(1+self.dataset.fake_atom_p/2))**2
        edges_per_sample = 3000 # manually computed the expectation of the square of number of atoms in training data, account for fake atoms
        samples_per_batch = edges_per_batch / edges_per_sample
        self.batches_per_epoch = self.samples_per_epoch // samples_per_batch
        self.batches_per_epoch = int(self.batches_per_epoch)
        

    def setup_queue(self):
        start_idx = int(self.frac_start * len(self.dataset))
        end_idx = int(self.frac_end * len(self.dataset))
        self.sample_queue = torch.randperm(len(self.dataset))[start_idx:end_idx]
        self.queue_idx = 0

    def get_next_batch(self):

        batch_idxs = []
        n_edges = 0
        while n_edges < self.edges_per_batch:
            idx = self.sample_queue[self.queue_idx]
            n_edges += self.dataset.n_edges_per_graph[idx]
            batch_idxs.append(idx)
            self.queue_idx += 1

            if self.queue_idx >= len(self.sample_queue):
                self.setup_queue()

        return batch_idxs


    def __iter__(self):

        self.setup_queue()
        for _ in range(self.batches_per_epoch):
            next_batch = self.get_next_batch()
            yield next_batch


    def __len__(self):
        return self.batches_per_epoch