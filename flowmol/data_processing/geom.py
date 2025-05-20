import torch
from torch.nn.functional import one_hot
from rdkit import Chem
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict
import functools

from multiprocessing import Pool

@dataclass
class MoleculeData:
    positions: Optional[torch.Tensor] = None
    atom_types: Optional[torch.Tensor] = None
    atom_charges: Optional[torch.Tensor] = None
    bond_types: Optional[torch.Tensor] = None
    bond_idxs: Optional[torch.Tensor] = None
    bond_order_counts: Optional[torch.Tensor] = None
    unique_valencies: Optional[torch.Tensor] = None
    failed: bool = False
    failure_mode: Optional[str] = None

@dataclass
class BatchMoleculeData:
    positions: List[torch.Tensor]
    atom_types: List[torch.Tensor]
    atom_charges: List[torch.Tensor]
    bond_types: List[torch.Tensor]
    bond_idxs: List[torch.Tensor]
    bond_order_counts: torch.Tensor
    unique_valencies: torch.Tensor
    n_mols: int
    failed_idxs: List[int]
    failure_counts: Dict[str, int]

def batch_molecule_data(mol_data: List[MoleculeData]):
    special_fields = ['bond_order_counts', 'unique_valencies', 'failed', 'failure_mode']
    fields_to_concat = [field for field in mol_data[0].__dataclass_fields__.keys() if field not in special_fields]

    data_to_concat = {field: [] for field in fields_to_concat}
    failed_idxs = []
    failure_counts = defaultdict(int)
    all_bond_order_counts = torch.zeros(4, dtype=torch.int64)
    all_unique_valencies = []
    n_mols = 0
    for idx, mol in enumerate(mol_data):

        # take care of failed molecules
        if mol.failed:
            failed_idxs.append(idx)
            failure_counts[mol.failure_mode] += 1
            continue

        n_mols += 1

        # collect all data that is just concatenated
        for field in fields_to_concat:
            data_to_concat[field].append(getattr(mol, field))

        # handle bond order counts and unique valencies, which are not concatenated
        all_unique_valencies.append(mol.unique_valencies)
        all_bond_order_counts += mol.bond_order_counts

        if len(all_unique_valencies) > 100:
            unique_valencies = torch.unique(torch.cat(all_unique_valencies, dim=0), dim=0)
            all_unique_valencies = [unique_valencies]

    all_unique_valencies = torch.unique(torch.cat(all_unique_valencies, dim=0), dim=0)
    
    output = BatchMoleculeData(
        bond_order_counts=all_bond_order_counts,
        unique_valencies=all_unique_valencies,
        failed_idxs=failed_idxs,
        failure_counts=failure_counts,
        n_mols=n_mols,
        **data_to_concat,
    )
    return output
        

class MoleculeFeaturizer():

    def __init__(self, 
                 atom_map: str, 
                 n_cpus=1,
                 explicit_aromaticity=False,
                 ):
        self.n_cpus = n_cpus
        self.atom_map = atom_map
        self.atom_map_dict = {atom: i for i, atom in enumerate(atom_map)}
        self.explicit_aromaticity = explicit_aromaticity

        if self.n_cpus == 1:
            self.pool = None
        else:
            self.pool = Pool(self.n_cpus)

        if 'H' in atom_map:
            self.explicit_hydrogens = True
        else:    
            self.explicit_hydrogens = False

    
    def featurize_molecules(self, molecules):
        failure_counts = defaultdict(int)

        process_func = functools.partial(featurize_molecule, 
                               atom_map_dict=self.atom_map_dict, 
                               explicit_hydrogens=self.explicit_hydrogens,
                               explicit_aromaticity=self.explicit_aromaticity
                               )

        if self.n_cpus == 1:
            results = [process_func(molecule) for molecule in molecules]
        else:
            # args = [(molecule, self.atom_map_dict) for molecule in molecules]
            # results = self.pool.starmap(featurize_molecule, args)
            results = self.pool.map(process_func, molecules)

        batch_data: BatchMoleculeData = batch_molecule_data(results)
        return batch_data



def featurize_molecule(molecule: Chem.rdchem.Mol, atom_map_dict: Dict[str, int], explicit_hydrogens=True, explicit_aromaticity=False) -> MoleculeData:

    # sanitize the molecule
    try:
        Chem.SanitizeMol(molecule)
    except Chem.SanitizeException as e:
        print(f"Sanitization failed for molecule {molecule.GetProp('_Name')}", flush=True)
        return MoleculeData(
            failed=True,
            failure_mode='sanitization'
        )

    # kekulize the molecule
    if not explicit_aromaticity:
        try:
            Chem.Kekulize(molecule, clearAromaticFlags=True)
        except Chem.KekulizeException as e:
            print(f"Kekulization failed for molecule {molecule.GetProp('_Name')}", flush=True)
            return MoleculeData(
                failed=True,
                failure_mode='kekulization'
            )

    # if explicit_hydrogens is False, remove all hydrogens from the molecule
    if not explicit_hydrogens:
        molecule = Chem.RemoveHs(molecule)

    num_fragments = len(Chem.GetMolFrags(molecule, sanitizeFrags=False))
    if num_fragments > 1:
        print(f"Fragmented molecule with {num_fragments} fragments", flush=True)
        return MoleculeData(
            failed=True,
            failure_mode='fragmentation'
        )

    # get positions
    positions = molecule.GetConformer().GetPositions()
    positions = torch.from_numpy(positions)

    # get atom elements as a string
    # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atom_types_idx = torch.zeros(molecule.GetNumAtoms()).long()
    atom_types_str = []
    atom_charges = torch.zeros_like(atom_types_idx)
    for i, atom in enumerate(molecule.GetAtoms()):
        try:
            atom_types_idx[i] = atom_map_dict[atom.GetSymbol()]
        except KeyError:
            print(f"Atom {atom.GetSymbol()} not in atom map", flush=True)
            return MoleculeData(
                failed=True,
                failure_mode='atom_map'
            )
        
        atom_charges[i] = atom.GetFormalCharge()
        atom_types_str.append(atom.GetSymbol())

    # get atom types as one-hot vectors
    atom_types = one_hot(atom_types_idx, num_classes=len(atom_map_dict)).bool()

    atom_charges = atom_charges.type(torch.int32)

    # get one-hot encoded of existing bonds only (no non-existing bonds)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(molecule, useBO=True))
    edge_index = adj.triu().nonzero().contiguous() # upper triangular portion of adjacency matrix

    # compute valencies
    if not explicit_aromaticity:
        valencies = adj.sum(dim=1)
        tcv = torch.stack([atom_types_idx, atom_charges, valencies], dim=1)
    else:
        # if our molecules have aromatic bonds
        n_arom_bonds = (adj == 1.5).sum(dim=1).int()
        non_arom_valencies = adj.sum(dim=1) - n_arom_bonds*1.5
        non_arom_valencies = non_arom_valencies.int()
        tcv = torch.stack([atom_types_idx, atom_charges, n_arom_bonds, non_arom_valencies], dim=1)
    unique_valencies = torch.unique(tcv, dim=0)

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

    return MoleculeData(
        positions=positions,
        atom_types=atom_types,
        atom_charges=atom_charges,
        bond_types=edge_attr,
        bond_idxs=edge_index,
        bond_order_counts=bond_order_counts,
        unique_valencies=unique_valencies,
        failed=False
    )