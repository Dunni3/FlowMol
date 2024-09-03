import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import dgl

from flowmol.data_processing.dataset import MoleculeDataset
from flowmol.data_processing.samplers import SameSizeMoleculeSampler, SameSizeDistributedMoleculeSampler

class MoleculeDataModule(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, dm_prior_config: dict, batch_size: int, num_workers: int = 0, distributed: bool = False, max_num_edges: int = 40000):
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
            self.train_dataset = MoleculeDataset('train', self.dataset_config, prior_config=self.prior_config)
            self.val_dataset = MoleculeDataset('val', self.dataset_config, prior_config=self.prior_config)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                collate_fn=dgl.batch, 
                                num_workers=self.num_workers)

        return dataloader
    
        # i wrote the following code under the assumption that we had to align predictions to the target, but i don't think this is true
        if self.x_subspace == 'se3-quotient':
            # if we are using the se3-quotient subspace, then we need to do same-size sampling so that we can efficiently compute rigid aligments during training
            if self.distributed:
                batch_sampler = SameSizeDistributedMoleculeSampler(self.train_dataset, batch_size=self.batch_size, max_num_edges=self.max_num_edges)
            else:
                batch_sampler = SameSizeMoleculeSampler(self.train_dataset, batch_size=self.batch_size, max_num_edges=self.max_num_edges)

            dataloader = DataLoader(dataset=self.train_dataset, batch_sampler=batch_sampler, collate_fn=dgl.batch, num_workers=self.num_workers)

        elif self.x_subspace == 'com-free':
            # if we are using the com-free subspace, then we don't need to do same-size sampling - and life is easier!
            dataloader = DataLoader(self.train_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    collate_fn=dgl.batch, 
                                    num_workers=self.num_workers)

                
        return dataloader
    
    def val_dataloader(self):
        dataloader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size*2, 
                                shuffle=True, 
                                collate_fn=dgl.batch, 
                                num_workers=self.num_workers)
        return dataloader

        if self.x_subspace == 'se3-quotient':
            # if we are using the se3-quotient subspace, then we need to do same-size sampling so that we can efficiently compute rigid aligments during training
            if self.distributed:
                batch_sampler = SameSizeDistributedMoleculeSampler(self.train_dataset, batch_size=self.batch_size*2)
            else:
                batch_sampler = SameSizeMoleculeSampler(self.train_dataset, batch_size=self.batch_size*2)

            dataloader = DataLoader(dataset=self.train_dataset, batch_sampler=batch_sampler, collate_fn=dgl.batch, num_workers=self.num_workers)

        elif self.x_subspace == 'com-free':
            # if we are using the com-free subspace, then we don't need to do same-size sampling - and life is easier!
            dataloader = DataLoader(self.train_dataset, 
                                    batch_size=self.batch_size*2, 
                                    shuffle=True, 
                                    collate_fn=dgl.batch, 
                                    num_workers=self.num_workers)
        return dataloader
