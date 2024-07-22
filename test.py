import argparse
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from models.flowmol import FlowMol
from analysis.molecule_builder import SampledMolecule
from analysis.metrics import SampleAnalyzer
from typing import List
from rdkit import Chem
from model_utils.load import read_config_file
import pickle
import math
import time

def parse_args():
    p = argparse.ArgumentParser(description='Testing Script')
    p.add_argument('--model_dir', type=Path, help='Path to model directory', default=None)
    p.add_argument('--checkpoint', type=Path, help='Path to checkpoint file', default=None)
    p.add_argument('--output_file', type=Path, help='Path to output file', default=None)

    p.add_argument('--n_mols', type=int, default=100, help='The number of molecules to generate.')
    p.add_argument('--n_atoms_per_mol', type=int, default=None, help="The number of atoms in every molecule. If None, the number of atoms will be sampled independently for each molecule from the training data distribution.")
    p.add_argument('--n_timesteps', type=int, default=20, help="Number of timesteps for integration via Euler's method")
    # p.add_argument('--visualize', action='store_true', help='Visualize the sampled trajectories')
    p.add_argument('--xt_traj', action='store_true', help='Save the x-t trajectory of the sampled molecules')
    p.add_argument('--ep_traj', action='store_true', help='Save the endpoint trajectory of the sampled molecules')
    p.add_argument('--metrics', action='store_true', help='Compute metrics on the sampled molecules')
    p.add_argument('--max_batch_size', type=int, default=128, help='Maximum batch size for sampling molecules')
    p.add_argument('--baseline_comparison', action='store_true', help='Whether these samples are for comparison to the baseline. If true, output format will be different.')

    p.add_argument('--stochasticity', type=float, default=None, help='Stochasticity for sampling molecules, only applies to models using CTMC')
    p.add_argument('--hc_thresh', type=float, default=None, help='High confidence threshold for purity sampling, only applies to models using CTMC')
    
    p.add_argument('--seed', type=int, default=None)

    args = p.parse_args()

    if args.model_dir is not None and args.checkpoint is not None:
        raise ValueError('only specify model_dir or checkpoint but not both')
    
    if args.model_dir is None and args.checkpoint is None:
        raise ValueError('must specify model_dir or checkpoint')

    if args.hc_thresh is not None:
        if args.hc_thresh < 0 or args.hc_thresh > 1:
            raise ValueError('hc_thresh must be on the interval [0, 1]')

    return args


if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # if trajectories are requested in either format, set visualize to True
    if args.xt_traj or args.ep_traj:
        visualize = True
    else:
        visualize = False

    # set seed
    if args.seed is not None:
        seed_everything(args.seed)

    # get model directory and checkpoint file
    if args.model_dir is not None:
        model_dir = args.model_dir
        checkpoint_file = args.model_dir / 'checkpoints' / 'last.ckpt'
    elif args.checkpoint is not None:
        model_dir = args.checkpoint.parent.parent
        checkpoint_file = args.checkpoint

    # load model
    model = FlowMol.load_from_checkpoint(checkpoint_file)

    # get config file
    config_file = model_dir / 'config.yaml'

    # read config file
    config = read_config_file(config_file)

    # set device to cuda:0 if available otherwise cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # move model to device
    model = model.to(device)

    # set model to eval mode
    model.eval()

    # compute the number of batches
    n_batches = math.ceil(args.n_mols / args.max_batch_size)

    molecules = []
    start = time.time()
    for batch_idx in range(n_batches):

        n_mols_needed = args.n_mols - len(molecules)
        batch_size = min(n_mols_needed, args.max_batch_size)

        if args.n_atoms_per_mol is None:
            batch_molecules: List[SampledMolecule]  = model.sample_random_sizes(
                batch_size, 
                device=device, 
                n_timesteps=args.n_timesteps, 
                xt_traj=args.xt_traj,
                ep_traj=args.ep_traj,
                stochasticity=args.stochasticity,
                high_confidence_threshold=args.hc_thresh)
        else:
            n_atoms = torch.full((batch_size,), args.n_atoms_per_mol, dtype=torch.long, device=device)
            batch_molecules: List[SampledMolecule] = model.sample(
                n_atoms, 
                device=device, 
                n_timesteps=args.n_timesteps, 
                xt_traj=args.xt_traj,
                ep_traj=args.ep_traj,
                stochasticity=args.stochasticity,
                high_confidence_threshold=args.hc_thresh)

        molecules.extend(batch_molecules)
    end = time.time()
    sampling_time = end - start

    # get output file
    if args.output_file is not None:
        output_file = args.output_file
    elif args.baseline_comparison:
        output_file = model_dir / 'samples' / f'{model_dir.name}_baseline_comparison.pkl'
    else:
        output_file = model_dir / 'samples' / 'sampled_mols.sdf'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # handle the case where we are sampling molecules for comparison to the baseline
    if args.baseline_comparison:
        print(f'Writing molecules to {output_file}')
        rdkit_mols = [ m.rdkit_mol for m in molecules ]
        with open(output_file, 'wb') as f:
            pickle.dump((rdkit_mols, sampling_time), f)
        exit()

    # compute metrics if necessary
    if args.metrics:
        processed_data_dir = config['dataset']['processed_data_dir']
        sample_analyzer = SampleAnalyzer(processed_data_dir=Path(processed_data_dir))
        metrics = sample_analyzer.analyze(molecules)

        # compute js-divergence of energies
        js_div = sample_analyzer.compute_energy_divergence(molecules)
        metrics['energy_js_div'] = js_div

        metrics_txt_file = output_file.parent / f'{output_file.stem}_metrics.txt'
        metrics_pkl_file = output_file.parent / f'{output_file.stem}_metrics.pkl'


        print(f'Writing metrics to {metrics_txt_file} and {metrics_pkl_file}')
        with open(metrics_txt_file, 'w') as f:
            for k, v in metrics.items():
                f.write(f'{k}: {v}\n')
        with open(metrics_pkl_file, 'wb') as f:
            pickle.dump(metrics, f)

    # check that output file is an sdf file
    if output_file.suffix != '.sdf':
        raise ValueError('output file must be an sdf file')
    
    if not visualize:
        # print the output_file
        print(f'Writing molecules to {output_file}')
        
        # write molecules to sdf file
        sdf_writer = Chem.SDWriter(str(output_file))
        sdf_writer.SetKekulize(False)
        for mol in molecules:
            rdkit_mol = mol.rdkit_mol
            if rdkit_mol is not None:
                sdf_writer.write(rdkit_mol)
        sdf_writer.close()
    else:
        print('Trajectories requested, writing a seprate output file for each molecule trajectory')
        for mol_idx, mol in enumerate(molecules):

            # write x-t trajectory if requested
            if args.xt_traj:
                mol_output_file = output_file.parent / f'{output_file.stem}_{mol_idx}_xt{output_file.suffix}'
                # print(f'Writing molecule {mol_idx} to {mol_output_file}')

                sdf_writer = Chem.SDWriter(str(mol_output_file))
                sdf_writer.SetKekulize(False)
                for traj_mol in mol.traj_mols:
                    try:
                        sdf_writer.write(traj_mol)
                    except Exception as e:
                        print(e)
                        continue
                sdf_writer.close()
            

            # write endpoint trajectory if requested
            if args.ep_traj:
                mol_output_file = output_file.parent / f'{output_file.stem}_{mol_idx}_ep{output_file.suffix}'
                # print(f'Writing molecule {mol_idx} to {mol_output_file}')

                sdf_writer = Chem.SDWriter(str(mol_output_file))
                sdf_writer.SetKekulize(False)
                for traj_mol in mol.ep_traj_mols:
                    try:
                        sdf_writer.write(traj_mol)
                    except Exception as e:
                        print(e)
                        continue
                sdf_writer.close()

        print(f'All molecules written to {output_file.parent}')
    

    