from typing import Dict, List
import torch
from torch.nn.functional import one_hot
from rdkit import Chem

from multiprocessing import Pool

def featurize_molecules(molecules: list, atom_map: List[str], n_cpus=1):


    atom_map_dict = {atom: i for i, atom in enumerate(atom_map)}


    all_positions, all_atom_types, all_atom_charges, all_bond_types, all_bond_idxs = [], [], [], [], []

    if n_cpus == 1:
        for molecule in molecules:
            positions, atom_types, atom_charges, bond_types, bond_idxs = featurize_molecule(molecule, atom_map_dict)
            all_positions.append(positions)
            all_atom_types.append(atom_types)
            all_atom_charges.append(atom_charges)
            all_bond_types.append(bond_types)
            all_bond_idxs.append(bond_idxs)
    else:
        with Pool(n_cpus) as pool:
            args = [(molecule, atom_map_dict) for molecule in molecules]
            results = pool.starmap(featurize_molecule, args)
            for positions, atom_types, atom_charges, bond_types, bond_idxs in results:
                all_positions.append(positions)
                all_atom_types.append(atom_types)
                all_atom_charges.append(atom_charges)
                all_bond_types.append(bond_types)
                all_bond_idxs.append(bond_idxs)

    # find molecules that failed to featurize and count them
    num_failed = 0
    failed_idxs = []
    for i in range(len(all_positions)):
        if all_positions[i] is None:
            num_failed += 1
            failed_idxs.append(i)

    # remove failed molecules
    all_positions = [pos for i, pos in enumerate(all_positions) if i not in failed_idxs]
    all_atom_types = [atom for i, atom in enumerate(all_atom_types) if i not in failed_idxs]
    all_atom_charges = [charge for i, charge in enumerate(all_atom_charges) if i not in failed_idxs]
    all_bond_types = [bond for i, bond in enumerate(all_bond_types) if i not in failed_idxs]
    all_bond_idxs = [idx for i, idx in enumerate(all_bond_idxs) if i not in failed_idxs]

    return all_positions, all_atom_types, all_atom_charges, all_bond_types, all_bond_idxs, num_failed



def featurize_molecule(molecule: Chem.rdchem.Mol, atom_map_dict: Dict[str, int]):

    # get positions
    positions = molecule.GetConformer().GetPositions()
    positions = torch.from_numpy(positions)

    # get atom elements as a string
    # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atom_types_idx = torch.zeros(molecule.GetNumAtoms(), dtype=int)
    atom_charges = torch.zeros_like(atom_types_idx)
    for i, atom in enumerate(molecule.GetAtoms()):
        try:
            atom_types_idx[i] = atom_map_dict[atom.GetSymbol()]
        except KeyError:
            print(f"Atom {atom.GetSymbol()} not in atom map", flush=True)
            return None, None, None, None, None
        
        atom_charges[i] = atom.GetFormalCharge()

    # get atom types and charges as one-hot vectors
    atom_types = one_hot(atom_types_idx, num_classes=len(atom_map_dict)).bool()
    atom_charges = one_hot(atom_charges + 2, num_classes=6).bool()


    # get one-hot encoded of existing bonds only (no non-existing bonds)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(molecule, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    bond_types = bond_types.long()
    edge_attr = one_hot(bond_types, num_classes=5).bool()
    # TODO: we are getting two edges for every bond .. not sure how we will use edge features / make edge predictions in the model.. deal with this later

    return positions, atom_types, atom_charges, edge_attr, edge_index