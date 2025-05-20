import argparse
import atexit
import json
import pickle
import signal
import sys
from pathlib import Path
from typing import List
from collections import defaultdict

import numpy as np
import torch
import tqdm
import yaml
from rdkit import Chem
from multiprocessing import Pool

from flowmol.data_processing.geom import MoleculeFeaturizer, BatchMoleculeData
from flowmol.utils.dataset_stats import compute_p_c_given_a

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_exit_handler(running_file: Path):
    def exit_handler(*args, **kwargs):
        running_file.unlink()
    return exit_handler

def setup_exit_handler(running_file: Path):

    if running_file.exists():
        print(f"Running file {running_file} already exists. Exiting.")
        sys.exit(0)

    # create running file
    running_file.touch()

    # remove running file when script exits
    atexit.register(get_exit_handler(running_file))

    # register exit handler on SIGTERM signal using signal module
    signal.signal(signal.SIGTERM, get_exit_handler(running_file))
    signal.signal(signal.SIGINT, get_exit_handler(running_file))

def parse_args():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(description='Process geometry')

    p.add_argument('split_file', type=Path, help='path to split file')
    p.add_argument('--config', type=Path, help='config file path')

    p.add_argument('--start_idx', type=int, default=0, help='start index')
    p.add_argument('--end_idx', type=int, default=np.inf, help='end index')

    p.add_argument('--n_cpus', type=int, default=1, help='number of cpus to use when computing partial charges for confomers')
    p.add_argument('--chunk_size', type=int, default=100, help='number of molecules to process at a time')

    p.add_argument('--overwrite', action='store_true', help='overwrite existing files')
    p.add_argument('--save_interval', type=int, default=5, help='number of molecules after which to save processed data')

    # p.add_argument('--dataset_size', type=int, default=None, help='number of molecules in dataset, only used to truncate dataset for debugging')

    args = p.parse_args()

    # check that start_idx is before end_idx
    if args.start_idx >= args.end_idx:
        raise ValueError(f"start_idx must be less than end_idx")

    return args

def build_kekulized_valency_dict(unique_valencies):
    atom_type_idxs, atom_charges, valencies = unique_valencies.unbind(dim=1)
    
    unique_valencies_dict = {}
    for atom_type_idx, charge, valency in zip(atom_type_idxs, atom_charges, valencies):
        atom_type_idx = int(atom_type_idx)

        atom_type = config['dataset']['atom_map'][atom_type_idx]

        charge = int(charge)
        valency = int(valency)

        if atom_type not in unique_valencies_dict:
            unique_valencies_dict[atom_type] = {}
        if charge not in unique_valencies_dict[atom_type]:
            unique_valencies_dict[atom_type][charge] = []
        unique_valencies_dict[atom_type][charge].append(valency)
    return unique_valencies_dict

def build_explicit_arom_valency_dict(unique_valencies):
    atom_type_idxs, atom_charges, n_arom_bonds, non_arom_valencies = unique_valencies.unbind(dim=1)

    unique_valencies_dict = {}
    for atom_type_idx, charge, n_arom_bonds, non_arom_valency in zip(atom_type_idxs, atom_charges, n_arom_bonds, non_arom_valencies):
        atom_type_idx = int(atom_type_idx)

        atom_type = config['dataset']['atom_map'][atom_type_idx]

        charge = int(charge)
        n_arom_bonds = int(n_arom_bonds)
        non_arom_valency = int(non_arom_valency)

        if atom_type not in unique_valencies_dict:
            unique_valencies_dict[atom_type] = {}
        if charge not in unique_valencies_dict[atom_type]:
            unique_valencies_dict[atom_type][charge] = []
        unique_valencies_dict[atom_type][charge].append((n_arom_bonds, non_arom_valency))
    return unique_valencies_dict



if __name__ == "__main__":

    args = parse_args()

    # start_idx is not 0 or end_idx is not np.inf, then raise NotImplementedError
    if args.start_idx != 0 or args.end_idx != np.inf:
        raise NotImplementedError(f"dataset chunking not yet implemented")

    # if the overwrite or save_interval flags are set, then raise NotImplementedError
    if args.overwrite or args.save_interval != 5:
        raise NotImplementedError(f"overwrite and save_interval flags not yet implemented")

    # load config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get number of conformers per molecule
    try:
        n_conformers = config['dataset']['confs_per_mol']
    except KeyError:
        n_conformers = None
    print(f'n_conformers set to {n_conformers}')


    # get processed data directory and create it if it doesn't exist
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    processed_data_dir.mkdir(exist_ok=True)

    output_dir = processed_data_dir 

    # determine output file name
    output_file = output_dir / f'{args.split_file.stem}_processed.pt'

    # get the raw data file
    raw_data_file = args.split_file

    # load the raw data
    with open(raw_data_file, 'rb') as f:
        raw_data = pickle.load(f)

    # determine if we are processing the entire dataset or just a subset
    if args.start_idx == 0 and args.end_idx == np.inf:
        full_dataset = True
        args.save_interval = np.inf
    else:
        full_dataset = False

    if full_dataset:
        output_dir = processed_data_dir
    else:
        output_dir = processed_data_dir / 'chunks'
        output_dir.mkdir(exist_ok=True)

    if not full_dataset:
        raise NotImplementedError(f"dataset chunking is a stupid idea, make multiprocessing work instead")

    # determine output file name
    if full_dataset:
        output_file = output_dir / f'{args.split_file.stem}_processed.pt'
    else:
        output_file = output_dir / f'{args.split_file.stem}_{args.start_idx}_{args.end_idx}.pt'

    # get the directory where we write files for currently running jobs
    running_dir = processed_data_dir / 'running'
    running_dir.mkdir(exist_ok=True)

    # get the running file for this job
    running_file = running_dir / output_file.name

    # setup exit handler
    setup_exit_handler(running_file)

    # combine all the conformers from all the molecules into a single list
    all_molecules = []
    all_smiles = []
    for molecule_chunk in raw_data:
        all_smiles.append(molecule_chunk[0])

        if n_conformers is not None:
            n_confs_this_mol = min(n_conformers, len(molecule_chunk[1]))
        else:
            n_confs_this_mol = len(molecule_chunk[1])
            
        for conformer in molecule_chunk[1][:n_confs_this_mol]:
            all_molecules.append(conformer)
    del raw_data

    # determine start_idx and end_idx for molecule processing
    if full_dataset:
        start_idx = 0
        end_idx = len(all_molecules)
    else:
        start_idx = args.start_idx
        end_idx = args.end_idx

    # get the molecules we are going to process
    molecules = all_molecules[start_idx:end_idx]
    all_smiles = all_smiles[start_idx:end_idx]

    dataset_size = config['dataset']['dataset_size']


    all_positions = []
    all_atom_types = []
    all_atom_charges = []
    all_bond_types = []
    all_bond_idxs = []
    all_bond_order_counts = torch.zeros(4, dtype=torch.int64)

    mol_featurizer = MoleculeFeaturizer(
        config['dataset']['atom_map'], 
        n_cpus=args.n_cpus,
        explicit_aromaticity=config['mol-fm'].get('explicit_aromaticity', False),
        )

    # molecules is a list of rdkit molecules.  now we create an iterator that yields sub-lists of molecules. we do this using itertools:
    chunk_iterator = chunks(molecules, args.chunk_size)
    n_chunks = len(molecules) // args.chunk_size + 1

    tqdm_iterator = tqdm.tqdm(chunk_iterator, desc='Featurizing molecules', total=n_chunks)
    failed_molecules_bar = tqdm.tqdm(desc="Failed Molecules", unit="molecules")

    # create a tqdm bar to report the total number of molecules processed
    total_molecules_bar = tqdm.tqdm(desc="Total Molecules", unit="molecules", total=len(molecules))

    failed_molecules = 0
    failure_counts = defaultdict(int)
    unique_valencies = None
    for molecule_chunk in tqdm_iterator:

        batch_data: BatchMoleculeData = mol_featurizer.featurize_molecules(molecule_chunk)

        num_failed = len(batch_data.failed_idxs)

        failed_molecules += num_failed
        failed_molecules_bar.update(num_failed)
        total_molecules_bar.update(batch_data.n_mols)
        for k, v in batch_data.failure_counts.items():
            failure_counts[k] += v

        all_positions.extend(batch_data.positions)
        all_atom_types.extend(batch_data.atom_types)
        all_atom_charges.extend(batch_data.atom_charges)
        all_bond_types.extend(batch_data.bond_types)
        all_bond_idxs.extend(batch_data.bond_idxs)
        all_bond_order_counts += batch_data.bond_order_counts

        if unique_valencies is None:
            unique_valencies = batch_data.unique_valencies
        else:
            unique_valencies = torch.unique(torch.cat((unique_valencies, batch_data.unique_valencies), dim=0), dim=0)

        # early stopping - a feature only used for debugging / creating small datasets
        if dataset_size is not None and len(all_positions) > dataset_size and full_dataset:
            break

    # get number of atoms in every data point
    n_atoms_list = [ x.shape[0] for x in all_positions ]
    n_bonds_list = [ x.shape[0] for x in all_bond_idxs ]

    # convert n_atoms_list and n_bonds_list to tensors
    n_atoms_list = torch.tensor(n_atoms_list)
    n_bonds_list = torch.tensor(n_bonds_list)

    # concatenate all_positions and all_features into single arrays
    all_positions = torch.concatenate(all_positions, dim=0)
    all_atom_types = torch.concatenate(all_atom_types, dim=0)
    all_atom_charges = torch.concatenate(all_atom_charges, dim=0)
    all_bond_types = torch.concatenate(all_bond_types, dim=0)
    all_bond_idxs = torch.concatenate(all_bond_idxs, dim=0)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's node features
    node_idx_array = torch.zeros((len(n_atoms_list), 2), dtype=torch.int32)
    node_idx_array[:, 1] = torch.cumsum(n_atoms_list, dim=0)
    node_idx_array[1:, 0] = node_idx_array[:-1, 1]

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's edge features
    edge_idx_array = torch.zeros((len(n_bonds_list), 2), dtype=torch.int32)
    edge_idx_array[:, 1] = torch.cumsum(n_bonds_list, dim=0)
    edge_idx_array[1:, 0] = edge_idx_array[:-1, 1]

    all_positions = all_positions.type(torch.float32)
    all_atom_charges = all_atom_charges.type(torch.int32)
    all_bond_idxs = all_bond_idxs.type(torch.int32)

    # create a dictionary to store all the data
    data_dict = {
        'smiles': all_smiles,
        'positions': all_positions,
        'atom_types': all_atom_types,
        'atom_charges': all_atom_charges,
        'bond_types': all_bond_types,
        'bond_idxs': all_bond_idxs,
        'node_idx_array': node_idx_array,
        'edge_idx_array': edge_idx_array,
    }

    # save the data
    torch.save(data_dict, output_file)

    # compute the marginal distribution of atom types, p(a)
    p_a = all_atom_types.sum(dim=0)
    p_a = p_a / p_a.sum()

    # compute the conditional distribution of charges given atom type, p(c|a)
    p_c_given_a = compute_p_c_given_a(all_atom_charges, all_atom_types, config['dataset']['atom_map'])

    # compute the marginal distribution of bond types, p(e)
    p_e = all_bond_order_counts / all_bond_order_counts.sum()

    # compute the marginal distirbution of charges, p(c)
    charge_vals, charge_counts = torch.unique(all_atom_charges, return_counts=True)
    p_c = torch.zeros(6, dtype=torch.float32)
    for c_val, c_count in zip(charge_vals, charge_counts):
        p_c[c_val+2] = c_count
    p_c = p_c / p_c.sum()

    # save p(a), p(e), p(c) and p(c|a) to a file
    marginal_dists_file = output_dir / f'{args.split_file.stem}_marginal_dists.pt'
    torch.save((p_a, p_c, p_e, p_c_given_a), marginal_dists_file)


    # create histogram of number of atoms
    if full_dataset:
        n_atoms, counts = torch.unique(n_atoms_list, return_counts=True)
        histogram_file = output_dir / f'{args.split_file.stem}_n_atoms_histogram.pt'
        torch.save((n_atoms, counts), histogram_file)


    # write all_smiles to its own file if we are processing the full dataset
    if full_dataset:
        smiles_file = output_dir / f'{args.split_file.stem}_smiles.pkl'
        with open(smiles_file, 'wb') as f:
            pickle.dump(all_smiles, f)

    # print failure counts to console and also write them to a file
    print(f"Failed to process {failed_molecules} molecules")
    for k, v in failure_counts.items():
        print(f"failure mode: {k}, {v} molecules")
    failure_counts_file = output_dir / f'{args.split_file.stem}_failure_counts.json'
    with open(failure_counts_file, 'w') as f:
        json.dump(failure_counts, f)

    # convert unique valencies to a dict representation
    if unique_valencies.shape[1] == 3:
        unique_valencies_dict = build_kekulized_valency_dict(unique_valencies)
    elif unique_valencies.shape[1] == 4:
        unique_valencies_dict = build_explicit_arom_valency_dict(unique_valencies)
    else:
        raise ValueError(f"unique valencies has shape {unique_valencies.shape}, expected 3 or 4")
    
    explicit_aromaticity = config['mol-fm'].get('explicit_aromaticity', False)
    if explicit_aromaticity:
        name_str = 'aromatic'
    else:
        name_str = 'kekulized'

    # write unique valencies to a file
    valencies_file = output_dir / f'{args.split_file.stem}_valencies_{name_str}.json'
    with open(valencies_file, 'w') as f:
        json.dump(unique_valencies_dict, f)



