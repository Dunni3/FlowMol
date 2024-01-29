import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch

from data_processing.dataset import MoleculeDataset


class MoleculeDataModule(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, distributed: bool = False):
        super().__init__()
        self.distributed = distributed
        self.dataset_config = dataset_config
        self.save_hyperparameters()

    def setup(self, stage: str):

        if stage == 'fit':
            self.train_dataset = MoleculeDataset('train', self.dataset_config)
            self.val_dataset = MoleculeDataset('val', self.dataset_config)

    def train_dataloader(self):
        raise NotImplementedError
    
    def val_dataloader(self):
        raise NotImplementedError