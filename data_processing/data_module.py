import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import dgl

from data_processing.dataset import MoleculeDataset
from data_processing.samplers import SameSizeMoleculeSampler, SameSizeDistributedMoleculeSampler

class MoleculeDataModule(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, prior_config: dict, batch_size: int, num_workers: int = 0, distributed: bool = False):
        super().__init__()
        self.distributed = distributed
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage: str):

        if stage == 'fit':
            self.train_dataset = MoleculeDataset('train', self.dataset_config)
            self.val_dataset = MoleculeDataset('val', self.dataset_config)

    def train_dataloader(self):
        # at one point in time, I wrote custom batch samplers for sampling batches of molecules which had the same number of atoms
        # but this ended up being unneccessary, leaving the code here for now just incase
        # if self.distributed:
        #     batch_sampler = SameSizeDistributedMoleculeSampler(self.train_dataset, batch_size=self.batch_size)
        # else:
        #     batch_sampler = SameSizeMoleculeSampler(self.train_dataset, batch_size=self.batch_size)

        # dataloader = DataLoader(dataset=self.train_dataset, batch_sampler=batch_sampler, collate_fn=dgl.batch, num_workers=self.num_workers)
        dataloader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                collate_fn=dgl.batch, 
                                num_workers=self.num_workers)
        

        
        
        return dataloader
    
    def val_dataloader(self):
        # this code is is commented for the same reason as described in the train_dataloader method
        # if self.distributed:
        #     batch_sampler = SameSizeDistributedMoleculeSampler(self.val_dataset, batch_size=self.batch_size)
        # else:
        #     batch_sampler = SameSizeMoleculeSampler(self.val_dataset, batch_size=self.batch_size)

        # dataloader = DataLoader(dataset=self.val_dataset, batch_sampler=batch_sampler, collate_fn=dgl.batch, num_workers=0)
        dataloader = DataLoader(self.val_dataset, 
                                batch_size=self.batch_size*2, 
                                shuffle=True, 
                                collate_fn=dgl.batch, 
                                num_workers=self.num_workers)
        return dataloader
