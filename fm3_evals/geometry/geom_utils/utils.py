from collections import defaultdict

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from rdkit.Chem import AllChem


def generate_canonical_key(*components):
    """
    Generate a canonical key for any molecular component (atoms, bonds).
    This works for angles, bond lengths, and torsions.
    """
    key1 = tuple(components)
    key2 = tuple(reversed(components))
    return min(key1, key2)


def is_valid(mol, verbose=False):
    """
    Validate a molecule for single fragment and successful sanitization.

    Args:
        mol (Chem.Mol): RDKit molecule object.
        verbose (bool): Print error messages if validation fails.

    Returns:
        bool: True if valid, otherwise False.
    """
    if mol is None:
        return False

    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException as e:
        if verbose:
            print(f"Kekulization failed: {e}")
        return False
    except ValueError as e:
        if verbose:
            print(f"Sanitization failed: {e}")
        return False

    if len(Chem.GetMolFrags(mol)) > 1:
        if verbose:
            print("Molecule has multiple fragments.")
        return False

    return True


def bond_type_to_symbol(bond_type_numeric):
    """Convert bond type numeric to chemical symbol."""
    if bond_type_numeric == 1:
        return "-"
    elif bond_type_numeric == 2:
        return "="
    elif bond_type_numeric == 3:
        return "#"
    elif bond_type_numeric == 12:
        return ":"
    else:
        return "?"


def compute_statistics(diff_sums):
    """
    Computes statistics: average difference, standard deviation, and weight.
    """
    total = 0
    for key, (diff_list, count) in diff_sums.items():
        total += count
    avg_diffs = {}
    for key, (diff_list, count) in diff_sums.items():
        avg_diff = np.mean(diff_list) if count > 0 else 0
        std_dev = np.std(diff_list) if count > 0 else 0
        weight = count / total if total > 0 else 0
        avg_diffs[key] = (avg_diff, std_dev, weight)
    return avg_diffs


def compute_differences(pairs, compute_function, show_progress=False):
    """
    Compute geometry differences using a specific `compute_function`.
    Optionally display a progress bar.

    Args:
        pairs (list): List of (init_mol, opt_mol) pairs.
        compute_function (function): Function that computes differences for a molecule pair.
        show_progress (bool): Whether to show a tqdm progress bar.

    Returns:
        dict: Dictionary with geometry difference stats.
    """
    diff_sums = defaultdict(lambda: [[], 0])
    iterator = tqdm(pairs, total=len(pairs), desc="Processing Molecules Sequentially") if show_progress else pairs

    results = []
    for pair in iterator:
        result = compute_function(pair)
        results.append(result)

    total_bonds = 0
    for result in results:
        for key, (diff_list, count) in result.items():
            diff_sums[key][0].extend(diff_list)
            diff_sums[key][1] += count
            total_bonds += count

    return compute_statistics(diff_sums)


def compute_rmsd(init_mol, opt_mol, hydrogens=True):
    """
    Compute the RMSD between the initial and optimized molecules by copying the
    conformer from opt_mol to a new molecule created from init_mol and aligning it.
    """
    init_mol.AddConformer(opt_mol.GetConformer(), assignId=True)
    if not hydrogens:
        init_mol = Chem.Mol(init_mol)
        init_mol = Chem.RemoveAllHs(init_mol)
    rmsd = AllChem.AlignMol(init_mol, init_mol, prbCid=0, refCid=1)
    return rmsd


def compute_mmff_energy_drop(mol, max_iters=1000):
    """
    Computes the MMFF energy drop of a molecule before and after MMFF optimization.

    Parameters:
    - mol: RDKit molecule with a conformer.
    - max_iters: Maximum number of optimization iterations.

    Returns:
    - energy_drop: The energy difference (E_before - E_after), or None if MMFF fails.
    """
    try:
        # Clone the molecule to preserve the original
        mol_copy = Chem.Mol(mol)

        # Compute initial MMFF energy
        props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant='MMFF94')
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
        e_before = ff.CalcEnergy()

        # Run MMFF geometry optimization
        success = AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
        if success != 0:
            return None  # Optimization failed

        # Compute energy after optimization
        ff_opt = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
        e_after = ff_opt.CalcEnergy()

        # Return energy drop
        return e_before - e_after

    except Exception as e:
        print(f"MMFF energy drop computation failed: {e}")
        return None


def compute_rmsd(init_mol, opt_mol, hydrogens=True):
    """Compute RMSD between initial and optimized molecule coordinates."""
    init_mol.AddConformer(opt_mol.GetConformer(), assignId=True)
    if not hydrogens:
        init_mol = Chem.RemoveAllHs(Chem.Mol(init_mol))
    return AllChem.AlignMol(init_mol, init_mol, prbCid=0, refCid=1)

