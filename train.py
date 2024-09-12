import argparse
from pathlib import Path
import yaml
import dgl
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import wandb
import dgl
import sys

# from models.ligand_edm import LigandEquivariantDiffusion
from flowmol.models.flowmol import FlowMol
from flowmol.data_processing.data_module import MoleculeDataModule
from flowmol.model_utils.sweep_config import merge_config_and_args, register_hyperparameter_args
from flowmol.model_utils.load import read_config_file, model_from_config, data_module_from_config

def parse_args():
    p = argparse.ArgumentParser(description='Training Script')
    p.add_argument('--config', type=Path, default=None)
    p.add_argument('--resume', type=Path, default=None, help='Path to run directory or checkpoint file to resume from')
    p.add_argument('--seed-model', type=Path, default=None, help='Path to a model checkpoint to seed the model with')

    # create a boolean argument for whether or not this is a debug run
    p.add_argument('--debug', action='store_true')

    p.add_argument('--seed', type=int, default=None)

    # create command line arguments for model hyperparameters
    register_hyperparameter_args(p)

    args = p.parse_args()

    if args.config is not None and args.resume is not None:
        raise ValueError('only specify a config file or a resume file but not both')

    return args


if __name__ == "__main__":
    args = parse_args()
    
    # TODO: implement resuming
    if args.resume is not None:
        # determine if we are resuming from a run directory or a checkpoint file
        if args.resume.is_dir():
            # we are resuming from a run directory
            # get the config file from the run directory
            run_dir = args.resume
            ckpt_file = str(run_dir / 'checkpoints' / 'last.ckpt')
        elif args.resume.is_file():
            run_dir = args.resume.parent.parent
            ckpt_file = str(args.resume)
        else:
            raise ValueError('resume argument must be a run directory or a checkpoint file that must already exist')
        
        config_file = run_dir / 'config.yaml'
    else:
        config_file = args.config
        ckpt_file = None


    # set seed
    if args.seed is not None:
        pl.seed_everything(args.seed)
    
    # process config file into dictionary
    config = read_config_file(config_file)

    # merge the config file with the command line arguments
    config = merge_config_and_args(config, args)
    
    # get wandb logger config
    wandb_config = config['wandb']

    # if this is a debug run, set the wandb logger mode to disabled
    if args.debug:
        wandb_config['mode'] = 'offline'
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
    output_dir = Path(config['training']['output_dir'])
    wandb_config['save_dir'] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # create wandb logger
    wandb_logger = WandbLogger(config=config, **wandb_config)
    # wandb_logger.experiment # not sure why this line is here...

    # save the config file to the run directory
    # include the run_id so we can resume this run later
    if args.resume is None and rank_zero_only.rank == 0:
        wandb_logger.experiment
        run_id = wandb_logger.experiment.id
        config['resume'] = {}
        config['resume']['run_id'] = run_id
        config['wandb']['name'] = wandb_logger.experiment.name
        run_dir = output_dir / f'{wandb_logger.experiment.name}_{run_id}'
        run_dir.mkdir(parents=True, exist_ok=True)
        print('Results are being written to: ', run_dir)
        with open(run_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

    # create ModelCheckpoint callback
    if rank_zero_only.rank == 0:
        checkpoints_dir = run_dir / 'checkpoints'
    else:
        checkpoints_dir = Path('/scr') / 'checkpoints'
    checkpoint_config = config['checkpointing']
    checkpoint_config['dirpath'] = str(checkpoints_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_config)

    # create data module
    data_module: MoleculeDataModule = data_module_from_config(config)

    # create model
    model: FlowMol = model_from_config(config, seed_ckpt=args.seed_model)

    # get pl trainer config
    trainer_config = config['training']['trainer_args']

    # compute the validation interval and add arguments for the pl.Trainer object accordingly
    trainer_config['val_check_interval'] = config['training']['evaluation']['val_loss_interval']
    trainer_config['check_val_every_n_epoch'] = 1

    # if this is a debug run, set limit_train_batches to 10
    if args.debug:
        trainer_config['limit_train_batches'] = 100

    trainer_config['use_distributed_sampler'] = True
        
    # set refresh rate for progress bar via TQDMProgressBar callback
    if args.debug:
        refresh_rate = 1
    else:
        refresh_rate = 20
    pbar_callback = TQDMProgressBar(refresh_rate=refresh_rate)

    trainer = pl.Trainer(logger=wandb_logger, **trainer_config, callbacks=[checkpoint_callback, pbar_callback])
    
    # train
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_file)
