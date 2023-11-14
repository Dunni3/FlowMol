import argparse
from pathlib import Path
import yaml
import torch
import dgl
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
import wandb
import dgl
import sys

# from models.ligand_edm import LigandEquivariantDiffusion
from models.mol_fm import MolFM
from data_processing.dataset import MoleculeDataset

def parse_args():
    p = argparse.ArgumentParser(description='Training Script')
    p.add_argument('--config', type=Path, default=None)
    p.add_argument('--resume', type=Path, default=None, help='path to checkpoint file')

    # TODO: make these arguments do something
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup_length', type=float, default=1)

    # create a boolean argument for whether or not this is a debug run
    p.add_argument('--debug', action='store_true')

    p.add_argument('--seed', type=int, default=42)


    args = p.parse_args()

    if args.config is not None and args.resume is not None:
        raise ValueError('only specify a config file or a resume file but not both')

    return args


if __name__ == "__main__":
    args = parse_args()
    
    # TODO: implement resuming
    if args.resume is not None:
        raise NotImplementedError

    # set seed
    pl.seed_everything(args.seed)
    
    # process config file into dictionary
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create dataset
    test_dataset = MoleculeDataset(split='test', dataset_config=config['dataset'])

    # create dataloader
    dataloader = DataLoader(test_dataset, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=True, 
                            collate_fn=dgl.batch, 
                            num_workers=config['training']['num_workers'])
    
    # get an example item from the dataset
    example_graph: dgl.DGLGraph = test_dataset[0]

    # get the filepath of the n_atoms histogram
    n_atoms_hist_filepath = Path(config['dataset']['processed_data_dir']) / 'n_atoms_histogram.pt'

    # create model
    n_atom_types = len(config['dataset']['atom_map'])
    model = MolFM(n_atom_types=n_atom_types, 
                  batches_per_epoch=len(dataloader), 
                  n_atoms_hist_file=n_atoms_hist_filepath,
                  vector_field_config=config['vector_field'],
                  interpolant_scheduler_config=config['interpolant_scheduler'], 
                  lr_scheduler_config=config['lr_scheduler'],
                  **config['mol_fm'])
    
    # get wandb logger config
    wandb_config = config['wandb']

    # if this is a debug run, set the wandb logger mode to disabled
    if args.debug:
        wandb_config['mode'] = 'disabled'
        wandb_config['name'] = 'debug_run'

    # if we are not resuming a run, generate a run_id
    if args.resume is None:
        run_id = wandb.util.generate_id()
        wandb_config['id'] = run_id
    else:
        # we are resuming a run, so get the run_id from the resume file
        run_id = config['resume']['run_id']
        wandb_config['id'] = run_id
        wandb_config['resume'] = 'must'

    # create the logging directory if it doesn't exist
    log_dir = Path(wandb_config['save_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # create wandb logger
    wandb_logger = WandbLogger(config=config, **wandb_config)

    # get run directory
    run_dir = Path(wandb.run.dir)
    checkpoints_dir = run_dir / 'checkpoints'

    # create ModelCheckpoint callback
    checkpoint_config = config['checkpointing']
    checkpoint_config['dirpath'] = str(checkpoints_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_config)

    # save the config file to the run directory
    # include the run_id so we can resume this run later
    if args.resume is None:
        config['resume'] = {}
        config['resume']['run_id'] = run_id
        with open(run_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

    # get pl trainer config
    trainer_config = config['training']['trainer_args']

    # if this is a debug run, set limit_train_batches to 10
    if args.debug:
        trainer_config['limit_train_batches'] = 10

    # create trainer
    trainer = pl.Trainer(logger=wandb_logger, **trainer_config, callbacks=[checkpoint_callback])
    
    # train
    trainer.fit(model, dataloader)
