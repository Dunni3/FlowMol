import argparse
from pathlib import Path
import yaml
import torch
from multiprocessing import Pool

from flowmol.data_processing.dataset import MoleculeDataset

# I created this script because I was planning on doing pre-alignment. That is, for every molecule in the dataset, 
# this script would sample the prior and compute the OT-alignment between the prior and the molecule.
# however, i think that this is not necessary, and that the alignment can be computed on the fly during training.
# leaving this script here for now, but it is not used in the current implementation.
# this script is also not complete, and would need to be modified to work with the current implementation.

def parse_args():
    p = argparse.ArgumentParser(description='Training Script')
    p.add_argument('--config', type=Path, default=None)
    p.add_argument('--split', type=str, default=None)

    # TODO: make these arguments do something
    p.add_argument('--n_cpus', type=int, default=1)

    p.add_argument('--seed', type=int, default=42)

def align_split(split: str, config: dict, n_cpus: int):

    dataset_config = config['dataset']
    processed_data_dir: Path = Path(dataset_config['processed_data_dir'])
    data_file = processed_data_dir / f'{split}_data_processed.pt'

    # load data from processed data directory
    data_dict = torch.load(data_file)

    # positions = data_dict['positions']
    # atom_types = data_dict['atom_types']
    # atom_charges = data_dict['atom_charges']
    # bond_types = data_dict['bond_types']
    # bond_idxs = data_dict['bond_idxs']
    node_idx_array = data_dict['node_idx_array']
    # edge_idx_array = data_dict['edge_idx_array']

    n_atoms = node_idx_array[:, 1] - node_idx_array[:, 0]
    n_atoms = n_atoms.tolist()
    positions_split = torch.split(data_dict['positions'], n_atoms)
    atom_types_split = torch.split(data_dict['atom_types'], n_atoms)
    atom_charges_split = torch.split(data_dict['atom_charges'], n_atoms)


    # construct arguments for computing optimal transport prior for each molecule
    compute_ot_args = []
    for dataset_idx, (mol_pos, mol_types, mol_charges) in enumerate(zip(positions_split, atom_types_split, atom_charges_split)):
        dst_dict = {
            'x': mol_pos,
            'a': mol_types,
            'c': mol_charges
        }
        compute_ot_args.append((dst_dict, dataset_idx))

    if n_cpus == 1:
        results = [compute_ot_prior(*args) for args in compute_ot_args]
    else:
        with Pool(n_cpus) as pool:
            results = pool.starmap(compute_ot_prior, compute_ot_args)

def compute_ot_prior(dst_dict, dataset_idx):
    prior_dict = {}
    cat_features = ['a', 'c']

    for feat in dst_dict.keys():

        # get destination features (t=1)
        dst_feat = dst_dict[feat]

if __name__ == "__main__":
    args = parse_args()

    # TODO: set seed

    if args.split is None:
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for split in splits:
        align_split(split, config, args.n_cpus)

