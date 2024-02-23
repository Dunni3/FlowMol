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
from multiprocessing import Pool
import pandas as pd

from data_processing.geom import MoleculeFeaturizer
from utils.dataset_stats import compute_p_c_given_a

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(description='Process geometry')
    p.add_argument('--config', type=Path, help='config file path')
    p.add_argument('--chunk_size', type=int, default=1000, help='number of molecules to process at once')

    p.add_argument('--n_cpus', type=int, default=1, help='number of cpus to use when computing partial charges for confomers')
    # p.add_argument('--dataset_size', type=int, default=None, help='number of molecules in dataset, only used to truncate dataset for debugging')

    args = p.parse_args()

    return args

def process_split(split_df, split_name, args, dataset_config):

    # get processed data directory and create it if it doesn't exist
    output_dir = Path(config['dataset']['processed_data_dir'])
    output_dir.mkdir(exist_ok=True) 

    raw_dir = Path(dataset_config['raw_data_dir']) 
    sdf_file = raw_dir / 'gdb9.sdf'
    bad_mols_file = raw_dir / 'uncharacterized.txt'

    # get the molecule ids to skip
    ids_to_skip = set()
    with open(bad_mols_file, 'r') as f:
        lines = f.read().split('\n')[9:-2]
        for x in lines:
            ids_to_skip.add(int(x.split()[0]) - 1)

    # get the molecule ids that are in our split
    mol_idxs_in_split = set(split_df.index.values.tolist())

    dataset_size = dataset_config['dataset_size']
    if dataset_size is None:
        dataset_size = np.inf

    # read all the molecules from the sdf file
    all_molecules = []
    all_smiles = []
    mol_reader = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
    for mol_idx, mol in enumerate(mol_reader):

        # skip molecules that are in the bad_mols_file or not in this split
        if mol_idx in ids_to_skip or mol_idx not in mol_idxs_in_split:
            continue

        all_molecules.append(mol)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smiles is not None:
            all_smiles.append(smiles)  # Convert mol to smiles string and append to all_smiles

        if len(all_molecules) > dataset_size:
            break


    all_positions = []
    all_atom_types = []
    all_atom_charges = []
    all_bond_types = []
    all_bond_idxs = []
    all_bond_order_counts = torch.zeros(5, dtype=torch.int64)

    mol_featurizer = MoleculeFeaturizer(config['dataset']['atom_map'], n_cpus=args.n_cpus)

    # molecules is a list of rdkit molecules.  now we create an iterator that yields sub-lists of molecules. we do this using itertools:
    chunk_iterator = chunks(all_molecules, args.chunk_size)
    n_chunks = len(all_molecules) // args.chunk_size + 1

    tqdm_iterator = tqdm.tqdm(chunk_iterator, desc='Featurizing molecules', total=n_chunks)
    failed_molecules_bar = tqdm.tqdm(desc="Failed Molecules", unit="molecules")

    # create a tqdm bar to report the total number of molecules processed
    total_molecules_bar = tqdm.tqdm(desc="Total Molecules", unit="molecules", total=len(all_molecules))

    failed_molecules = 0
    for molecule_chunk in tqdm_iterator:

        # TODO: we should collect all the molecules from each individual list into a single list and then featurize them all at once - this would make the multiprocessing actually useful
        positions, atom_types, atom_charges, bond_types, bond_idxs, num_failed, bond_order_counts = mol_featurizer.featurize_molecules(molecule_chunk)

        failed_molecules += num_failed
        failed_molecules_bar.update(num_failed)
        total_molecules_bar.update(len(molecule_chunk))

        all_positions.extend(positions)
        all_atom_types.extend(atom_types)
        all_atom_charges.extend(atom_charges)
        all_bond_types.extend(bond_types)
        all_bond_idxs.extend(bond_idxs)
        all_bond_order_counts += bond_order_counts

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

    # determine output file name and save the data_dict there
    output_file = output_dir / f'{split_name}_processed.pt'
    torch.save(data_dict, output_file)

    # create histogram of number of atoms
    n_atoms, counts = torch.unique(n_atoms_list, return_counts=True)
    histogram_file = output_dir / f'{split_name}_n_atoms_histogram.pt'
    torch.save((n_atoms, counts), histogram_file)

    # compute the marginal distribution of atom types, p(a)
    p_a = all_atom_types.sum(dim=0)
    p_a = p_a / p_a.sum()

    # compute the marginal distribution of bond types, p(e)
    p_e = all_bond_order_counts / all_bond_order_counts.sum()

    # compute the marginal distirbution of charges, p(c)
    charge_vals, charge_counts = torch.unique(all_atom_charges, return_counts=True)
    p_c = torch.zeros(6, dtype=torch.float32)
    for c_val, c_count in zip(charge_vals, charge_counts):
        p_c[c_val+2] = c_count
    p_c = p_c / p_c.sum()

    # compute the conditional distribution of charges given atom type, p(c|a)
    p_c_given_a = compute_p_c_given_a(all_atom_charges, all_atom_types, dataset_config['atom_map'])

    # save p(a), p(e) and p(c|a) to a file
    marginal_dists_file = output_dir / f'{split_name}_marginal_dists.pt'
    torch.save((p_a, p_c, p_e, p_c_given_a), marginal_dists_file)

    # write all_smiles to its own file
    smiles_file = output_dir / f'{split_name}_smiles.pkl'
    with open(smiles_file, 'wb') as f:
        pickle.dump(all_smiles, f)


if __name__ == "__main__":

    # parse command-line args
    args = parse_args()

    # load config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_config = config['dataset']
    if dataset_config['dataset_name'] != 'qm9':
        raise ValueError('This script only works with the qm9 dataset')

    ##########3
    # this must be changed for the qm9 dataset
    ############3

    # get qm9 csv file as a pandas dataframe
    qm9_csv_file = Path(dataset_config['raw_data_dir']) / 'gdb9.sdf.csv'
    df = pd.read_csv(qm9_csv_file)

    n_samples = df.shape[0]
    n_train = 100000
    n_test = int(0.1 * n_samples)
    n_val = n_samples - (n_train + n_test)

    # print the number of samples in each split
    print(f"Number of samples in train split: {n_train}")
    print(f"Number of samples in test split: {n_test}")
    print(f"Number of samples in val split: {n_val}")

    # Shuffle dataset with df.sample, then split
    train, val, test = np.split(df.sample(frac=1, random_state=42), [n_train, n_val + n_train])
    

    split_names = ['train_data', 'val_data', 'test_data']
    for split_df, split_name in zip([train, val, test], split_names):
        process_split(split_df, split_name, args, dataset_config)

    




