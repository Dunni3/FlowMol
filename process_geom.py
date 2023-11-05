import argparse
import atexit
import json
import pickle
import signal
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import tqdm
import yaml
from rdkit import Chem

from data_processing.geom import featurize_molecules


def parse_args():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(description='Process geometry')

    p.add_argument('split_file', type=Path, help='path to split file')
    p.add_argument('--config', type=Path, help='config file path')

    p.add_argument('--start_idx', type=int, default=0, help='start index')
    p.add_argument('--end_idx', type=int, default=np.inf, help='end index')

    p.add_argument('--n_cpus', type=int, default=1, help='number of cpus to use when computing partial charges for confomers')

    p.add_argument('--overwrite', action='store_true', help='overwrite existing files')
    p.add_argument('--save_interval', type=int, default=5, help='number of molecules after which to save processed data')

    p.add_argument('--dataset_size', type=int, default=None, help='number of molecules in dataset, only used to truncate dataset for debugging')

    args = p.parse_args()

    # check that start_idx is before end_idx
    if args.start_idx >= args.end_idx:
        raise ValueError(f"start_idx must be less than end_idx")

    return args

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


    all_smiles = []
    all_positions = []
    all_atom_types = []
    all_atom_charges = []
    all_bond_types = []
    all_bond_idxs = []

    tqdm_iterator = tqdm.tqdm(raw_data, desc='Featurizing molecules', total=len(raw_data))
    failed_molecules_bar = tqdm.tqdm(desc="Failed Molecules", unit="molecules")

    failed_molecules = 0
    for molecule_list in tqdm_iterator:

        # molecule_list is a list (smiles_string, *conformers as rdkit objects)
        smiles_string = molecule_list[0]
        conformers = molecule_list[1]

        # TODO: we should collect all the molecules from each individual list into a single list and then featurize them all at once - this would make the multiprocessing actually useful
        positions, atom_types, atom_charges, bond_types, bond_idxs, num_failed = featurize_molecules(conformers, atom_map=config['dataset']['atom_map'], n_cpus=args.n_cpus)

        failed_molecules += num_failed
        failed_molecules_bar.update(num_failed)

        all_smiles.append(smiles_string)
        all_positions.extend(positions)
        all_atom_types.extend(atom_types)
        all_atom_charges.extend(atom_charges)
        all_bond_types.extend(bond_types)
        all_bond_idxs.extend(bond_idxs)

        if args.dataset_size is not None and len(all_positions) > args.dataset_size:
            break

    # get number of atoms in every data point
    n_atoms_list = [ x.shape[0] for x in all_positions ]
    n_bonds_list = [ x.shape[0] for x in all_bond_idxs ]

    # convert n_atoms_list and n_bonds_list to tensors
    n_atoms_list = torch.tensor(n_atoms_list)
    n_bonds_list = torch.tensor(n_bonds_list)

    # concatenate all_positions and all_features into single arrays
    all_positions = torch.concatenate(all_positions, axis=0)
    all_atom_types = torch.concatenate(all_atom_types, axis=0)
    all_atom_charges = torch.concatenate(all_atom_charges, axis=0)
    all_bond_types = torch.concatenate(all_bond_types, axis=0)
    all_bond_idxs = torch.concatenate(all_bond_idxs, axis=0)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's node features
    node_idx_array = np.zeros((len(n_atoms_list), 2), dtype=int)
    node_idx_array[:, 1] = torch.cumsum(n_atoms_list, dim=0)
    node_idx_array[1:, 0] = node_idx_array[:-1, 1]

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's edge features
    edge_idx_array = torch.zeros((len(n_bonds_list), 2), dtype=int)
    edge_idx_array[:, 1] = torch.cumsum(n_bonds_list, dim=0)
    edge_idx_array[1:, 0] = edge_idx_array[:-1, 1]

    all_positions = all_positions.type(torch.float32)
    all_atom_charges = all_atom_charges.type(torch.uint8)
    all_bond_idxs = all_bond_idxs.type(torch.uint8)

    # create a dictionary to store all the data
    data_dict = {
        'smiles': all_smiles,
        'positions': all_positions,
        'atom_types': all_atom_types,
        'atom_charges': all_atom_charges,
        'bond_idxs': all_bond_idxs,
        'node_idx_array': node_idx_array,
        'edge_idx_array': edge_idx_array,
    }

    # save the data
    torch.save(data_dict, output_file)


