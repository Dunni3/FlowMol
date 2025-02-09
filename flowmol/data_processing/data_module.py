import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import dgl

from flowmol.data_processing.dataset import MoleculeDataset
from flowmol.data_processing.adaptive_sampler import AdaptiveEdgeSampler

class MoleculeDataModule(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, dm_prior_config: dict, batch_size: int, 
                 num_workers: int = 0, distributed: bool = False, 
                 max_num_edges: int = None):
        super().__init__()
        self.distributed = distributed
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prior_config = dm_prior_config
        self.max_num_edges = max_num_edges
        self.save_hyperparameters()

    def setup(self, stage: str):

        if stage == 'fit':
            self.train_dataset = self.load_dataset('train')
            self.val_dataset = self.load_dataset('val')

    def load_dataset(self, dataset_name: str):
        return MoleculeDataset(dataset_name, self.dataset_config, prior_config=self.prior_config)

    def train_dataloader(self):

        if self.max_num_edges is not None:
            batch_sampler = AdaptiveEdgeSampler(self.train_dataset, self.max_num_edges, distributed=self.distributed)
            batch_kwargs = {
                'batch_sampler':batch_sampler,
            }
        else:
            batch_kwargs = {
                'batch_size': self.batch_size,
                'shuffle': True,
            }


        dataloader = DataLoader(self.train_dataset, 
                                collate_fn=dgl.batch, 
                                num_workers=self.num_workers,
                                **batch_kwargs)

        return dataloader
    
    def val_dataloader(self):

        if self.max_num_edges is not None:
            batch_sampler = AdaptiveEdgeSampler(self.val_dataset, self.max_num_edges, distributed=self.distributed)
            batch_kwargs = {
                'batch_sampler': batch_sampler,
            }
        else:
            batch_kwargs = {
                'batch_size': self.batch_size * 2,
                'shuffle': True,
            }

        dataloader = DataLoader(self.val_dataset, 
                                collate_fn=dgl.batch, 
                                num_workers=self.num_workers,
                                **batch_kwargs)

        return dataloader
