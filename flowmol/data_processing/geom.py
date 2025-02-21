from typing import Dict, List
import torch
from torch.nn.functional import one_hot
from rdkit import Chem
from collections import defaultdict

from multiprocessing import Pool

class MoleculeFeaturizer():

    def __init__(self, atom_map: str, n_cpus=1):
        self.n_cpus = n_cpus
        self.atom_map = atom_map
        self.atom_map_dict = {atom: i for i, atom in enumerate(atom_map)}

        if self.n_cpus == 1:
            self.pool = None
        else:
            self.pool = Pool(self.n_cpus)

        if 'H' in atom_map:
            self.explicit_hydrogens = True
        else:    
            self.explicit_hydrogens = False

    def parse_result(self, result, failure_counts):

        if not isinstance(result, tuple):
            failure_counts[result] += 1
            positions = None
            atom_types = None
            atom_charges = None
            bond_types = None
            bond_idxs = None
            bond_order_counts = None
        else:
            positions, atom_types, atom_charges, bond_types, bond_idxs, bond_order_counts = result
        
        return positions, atom_types, atom_charges, bond_types, bond_idxs, bond_order_counts

    def featurize_molecules(self, molecules):


        all_positions, all_atom_types, all_atom_charges, all_bond_types, all_bond_idxs = [], [], [], [], []
        all_bond_order_counts = torch.zeros(4, dtype=torch.int64)

        failure_counts = defaultdict(int)

        if self.n_cpus == 1:
            for molecule in molecules:
                result = featurize_molecule(molecule, self.atom_map_dict)

                positions, atom_types, atom_charges, bond_types, bond_idxs, bond_order_counts = \
                self.parse_result(result, failure_counts)

                all_positions.append(positions)
                all_atom_types.append(atom_types)
                all_atom_charges.append(atom_charges)
                all_bond_types.append(bond_types)
                all_bond_idxs.append(bond_idxs)

                if bond_order_counts is not None:
                    all_bond_order_counts += bond_order_counts

        else:
            args = [(molecule, self.atom_map_dict) for molecule in molecules]
            results = self.pool.starmap(featurize_molecule, args)
            for result in results:
                positions, atom_types, atom_charges, bond_types, bond_idxs, bond_order_counts = \
                self.parse_result(result, failure_counts)
                all_positions.append(positions)
                all_atom_types.append(atom_types)
                all_atom_charges.append(atom_charges)
                all_bond_types.append(bond_types)
                all_bond_idxs.append(bond_idxs)

                if bond_order_counts is not None:
                    all_bond_order_counts += bond_order_counts

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

        return all_positions, all_atom_types, all_atom_charges, all_bond_types, all_bond_idxs, num_failed, all_bond_order_counts, failure_counts



def featurize_molecule(molecule: Chem.rdchem.Mol, atom_map_dict: Dict[str, int], explicit_hydrogens=True):

    # kekulize the molecule
    try:
        Chem.Kekulize(molecule)
    except Chem.KekulizeException as e:
        print(f"Kekulization failed for molecule {molecule.GetProp('_Name')}", flush=True)
        return 'kekulization'

    # if explicit_hydrogens is False, remove all hydrogens from the molecule
    if not explicit_hydrogens:
        molecule = Chem.RemoveHs(molecule)

    num_fragments = len(Chem.GetMolFrags(molecule, sanitizeFrags=False))
    if num_fragments > 1:
        print(f"Fragmented molecule with {num_fragments} fragments", flush=True)
        return 'fragmented'

    # get positions
    positions = molecule.GetConformer().GetPositions()
    positions = torch.from_numpy(positions)

    # get atom elements as a string
    # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atom_types_idx = torch.zeros(molecule.GetNumAtoms()).long()
    atom_charges = torch.zeros_like(atom_types_idx)
    for i, atom in enumerate(molecule.GetAtoms()):
        try:
            atom_types_idx[i] = atom_map_dict[atom.GetSymbol()]
        except KeyError:
            print(f"Atom {atom.GetSymbol()} not in atom map", flush=True)
            return 'atom_map'
        
        atom_charges[i] = atom.GetFormalCharge()

    # get atom types as one-hot vectors
    atom_types = one_hot(atom_types_idx, num_classes=len(atom_map_dict)).bool()

    atom_charges = atom_charges.type(torch.int32)

    # get one-hot encoded of existing bonds only (no non-existing bonds)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(molecule, useBO=True))
    edge_index = adj.triu().nonzero().contiguous() # upper triangular portion of adjacency matrix

    # note that because we take the upper-triangular portion of the adjacency matrix, there is only one edge per bond
    # at training time for every edge (i,j) in edge_index, we will also add edges (j,i)
    # we also only retain existing bonds, but then at training time we will add in edges for non-existing bonds

    bond_types = adj[edge_index[:, 0], edge_index[:, 1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.type(torch.int32)
    # edge_attr = one_hot(bond_types, num_classes=5).bool() # five bond classes: no bond, single, double, triple, aromatic

    # count the number of pairs of atoms which are bonded
    n_bonded_pairs = edge_index.shape[0]

    # compute the number of upper-edge pairs
    n_atoms = atom_types.shape[0]
    n_pairs = n_atoms * (n_atoms - 1) // 2

    # compute the number of pairs of atoms which are not bonded
    n_unbonded = n_pairs - n_bonded_pairs

    # construct an array containing the counts of each bond type in the molecule
    bond_order_idxs, existing_bond_order_counts = torch.unique(edge_attr, return_counts=True)
    bond_order_counts = torch.zeros(4, dtype=torch.int64)
    for bond_order_idx, count in zip(bond_order_idxs, existing_bond_order_counts):
        bond_order_counts[bond_order_idx] = count

    bond_order_counts[0] = n_unbonded

    return positions, atom_types, atom_charges, edge_attr, edge_index, bond_order_counts