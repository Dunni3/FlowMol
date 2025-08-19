import torch
from rdkit import Chem

from geom_utils.geom_drugs_valency_table import geom_drugs_h_tuple_valencies
from geom_utils.utils import is_valid


def _is_valid_valence_tuple(combo, allowed, charge):
    if isinstance(allowed, tuple):
        return combo == allowed
    elif isinstance(allowed, (list, set)):
        return combo in allowed
    elif isinstance(allowed, dict):
        return _is_valid_valence_tuple(combo, allowed.get(charge, []), charge)
    return False


def compute_molecules_stability_from_graph(adjacency_matrices, numbers, charges, allowed_bonds=None,
        aromatic=True):
    if adjacency_matrices.ndim == 2:
        adjacency_matrices = adjacency_matrices.unsqueeze(0)
        numbers = numbers.unsqueeze(0)
        charges = charges.unsqueeze(0)

    if allowed_bonds is None:
        allowed_bonds = geom_drugs_h_tuple_valencies

    if not aromatic:
        assert (adjacency_matrices == 1.5).sum() == 0 and (adjacency_matrices == 4).sum() == 0

    batch_size = adjacency_matrices.shape[0]
    stable_mask = torch.zeros(batch_size)
    n_stable_atoms = torch.zeros(batch_size)
    n_atoms = torch.zeros(batch_size)

    for i in range(batch_size):
        adj = adjacency_matrices[i]
        atom_nums = numbers[i]
        atom_charges = charges[i]

        mol_stable = True
        n_atoms_i, n_stable_i = 0, 0

        for j, (a_num, charge) in enumerate(zip(atom_nums, atom_charges)):
            if a_num.item() == 0:
                continue
            row = adj[j]
            aromatic_count = int((row == 1.5).sum().item())
            normal_valence = float((row * (row != 1.5)).sum().item())
            combo = (aromatic_count, int(normal_valence))
            symbol = Chem.GetPeriodicTable().GetElementSymbol(int(a_num))
            allowed = allowed_bonds.get(symbol, {})

            if _is_valid_valence_tuple(combo, allowed, int(charge)):
                n_stable_i += 1
            else:
                mol_stable = False

            n_atoms_i += 1

        stable_mask[i] = float(mol_stable)
        n_stable_atoms[i] = n_stable_i
        n_atoms[i] = n_atoms_i

    return stable_mask, n_stable_atoms, n_atoms


def compute_molecules_stability(rdkit_molecules, aromatic=True, allowed_bonds=None):
    stable_list, stable_atoms_list, atom_counts_list, validity_list = [], [], [], []

    for mol in rdkit_molecules:
        if mol is None:
            continue
        n_atoms = mol.GetNumAtoms()
        adj = torch.zeros((1, n_atoms, n_atoms))
        numbers = torch.zeros((1, n_atoms), dtype=torch.long)
        charges = torch.zeros((1, n_atoms), dtype=torch.long)

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            numbers[0, idx] = atom.GetAtomicNum()
            charges[0, idx] = atom.GetFormalCharge()

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            adj[0, i, j] = adj[0, j, i] = bond_type

        stable, stable_atoms, atom_count = compute_molecules_stability_from_graph(
            adj, numbers, charges, allowed_bonds, aromatic
        )
        stable_list.append(stable.item())
        stable_atoms_list.append(stable_atoms.item())
        atom_counts_list.append(atom_count.item())
        validity_list.append(float(is_valid(mol)))

    return (
        torch.tensor(validity_list),
        torch.tensor(stable_list),
        torch.tensor(stable_atoms_list),
        torch.tensor(atom_counts_list)
    )
