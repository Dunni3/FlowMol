import argparse
from pathlib import Path
import yaml
import torch
import itertools
import pickle

def parse_args():
    p = argparse.ArgumentParser(description='Combine chunks of the geom dataset produced by process_geom.py')

    p.add_argument('chunk_dir', type=Path, help='filepath to directory containing chunks to combine')
    p.add_argument('--config', type=Path, help='config file path')
    

    p.add_argument('--print_interval', type=int, default=100, help='print status every print_interval iterations')

    args = p.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # process config file into dictionary
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get processed data directory
    data_dir: Path = Path(config['dataset']['processed_data_dir'])

    # get chunk directory
    chunk_dir = args.chunk_dir

    # count the number of files in chunk_dir
    n_files = len(list(chunk_dir.iterdir()))

    # iterate over chunks
    smiles, positions, atom_types, atom_charges, bond_idxs, node_idx_array, edge_idx_array = [], [], [], [], [], [], []
    for idx, chunk_file in enumerate(chunk_dir.iterdir()):

        # print status
        if idx % args.print_interval == 0:
            # print the percentage of chunks processed
            print(f'{idx / n_files * 100:.2f}% complete', flush=True)

        # load chunk
        # chunk_positions, chunk_features, chunk_idxs, chunk_elements, _ = torch.load(chunk_file)
        data_dict = torch.load(chunk_file)
        chunk_positions = data_dict['positions']
        chunk_smiles = data_dict['smiles']
        chunk_atom_types = data_dict['atom_types']
        chunk_atom_charges = data_dict['atom_charges']
        chunk_bond_idxs = data_dict['bond_idxs']
        chunk_node_idx_array = data_dict['node_idx_array']
        chunk_edge_idx_array = data_dict['edge_idx_array']

        # append to list
        smiles.extend(chunk_smiles)
        positions.append(chunk_positions)
        atom_types.append(chunk_atom_types)
        atom_charges.append(chunk_atom_charges)
        bond_idxs.append(chunk_bond_idxs)
        node_idx_array.append(chunk_node_idx_array)
        edge_idx_array.append(chunk_edge_idx_array)

    # get number of atoms and edges in each chunk
    chunk_n_atoms = [ x.shape[0] for x in positions]
    chunk_n_atoms = torch.tensor(chunk_n_atoms)
    chunk_n_edges = [ x.shape[1] for x in bond_idxs]
    chunk_n_edges = torch.tensor(chunk_n_edges)

    # get number of atoms and bonds preceding each chunk
    n_atoms_preceding = [0] + torch.cumsum(chunk_n_atoms, dim=0)[:-1].tolist()
    n_atoms_preceding = torch.tensor(n_atoms_preceding)
    n_edges_preceding = [0] + torch.cumsum(chunk_n_edges, dim=0)[:-1].tolist()
    n_edges_preceding = torch.tensor(n_edges_preceding)

    # get number of molceules in each chunk
    chunk_n_molecules = [ idx_arr.shape[0] for idx_arr in node_idx_array]
    chunk_n_molecules = torch.tensor(chunk_n_molecules)

    # get chunk index of each molecule
    n_chunks = len(positions)
    molecule_chunk_idxs = torch.arange(n_chunks).repeat_interleave(chunk_n_molecules)


    # concatenate chunks
    positions = torch.cat(positions, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    atom_charges = torch.cat(atom_charges, dim=0)
    bond_idxs = torch.cat(bond_idxs, dim=1)
    node_idx_array = torch.cat(node_idx_array, dim=0)
    edge_idx_array = torch.cat(edge_idx_array, dim=0)

    # get the number of atoms in every molecule and the number of edges in every molecule
    atoms_per_molecule = node_idx_array[:, 1] - node_idx_array[:, 0]
    edges_per_molecule = edge_idx_array[:, 1] - edge_idx_array[:, 0]

    # get the number of occurences of each value in atoms_per_molecule and edges_per_molecule
    n_atoms, n_atoms_counts = torch.unique(atoms_per_molecule, return_counts=True)
    n_edges, n_edges_counts = torch.unique(edges_per_molecule, return_counts=True)

    # update the node idxs to be global idxs
    node_idx_array = node_idx_array + n_atoms_preceding[molecule_chunk_idxs].unsqueeze(-1)

    # update the edge idxs to be global idxs
    edge_idx_array = edge_idx_array + n_edges_preceding[molecule_chunk_idxs].unsqueeze(-1)

    # save positions, features, and idxs to file
    data_dict = {
        'smiles': smiles,
        'positions': positions,
        'atom_types': atom_types,
        'atom_charges': atom_charges,
        'bond_idxs': bond_idxs,
        'node_idx_array': node_idx_array,
        'edge_idx_array': edge_idx_array,
    }
    output_file_name = f"{chunk_file.stem.split('_')[0]}_processed.pt"
    torch.save(data_dict, data_dir / output_file_name)

    # save the histogram of the number of atoms in each molecule
    torch.save((n_atoms, n_atoms_counts), data_dir / '{output_file_name}_n_atoms_histogram.pt')


    # TODO: save all smiles as separate file, save edge histogram as separate file