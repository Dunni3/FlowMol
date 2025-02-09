import torch
from torch.utils.data import Sampler

from flowmol.data_processing.dataset import MoleculeDataset

class AdaptiveEdgeSampler(Sampler):
    def __init__(self, dataset, edges_per_batch: int,
                 distributed: bool = False,
                 rank: int = None,
                 num_replicas: int = None,
                  ):

        self.dataset: MoleculeDataset = dataset
        self.edges_per_batch = edges_per_batch

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
        

    def setup_queue(self):
        self.sample_queue = torch.randperm(len(self.dataset))
        self.queue_idx = 0
        if self.samples_per_epoch < len(self.sample_queue):
            self.sample_queue = self.sample_queue[:self.samples_per_epoch]

    def get_next_batch(self):

        batch_idxs = []
        n_edges = 0
        while n_edges < self.edges_per_batch:
            idx = self.sample_queue[self.queue_idx]
            n_edges += self.dataset.n_edges_per_graph[idx]
            batch_idxs.append(idx)
            self.queue_idx += 1
            if self.queue_idx >= len(self.sample_queue):
                break
        return batch_idxs


    def __iter__(self):

        self.setup_queue()
        while True:
            yield self.get_next_batch()
            if self.queue_idx >= len(self.sample_queue):
                break


    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size