from typing import List
from .molecule_builder import SampledMolecule
import torch
from rdkit import Chem

allowed_bonds = {'H': {0: 1, 1: 0, -1: 0},
                 'C': {0: [3, 4], 1: 3, -1: 3},
                 'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},    # In QM9, N+ seems to be present in the form NH+ and NH2+
                 'O': {0: 2, 1: 3, -1: 1},
                 'F': {0: 1, -1: 0},
                 'B': 3, 'Al': 3, 'Si': 4,
                 'P': {0: [3, 5], 1: 4},
                 'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                 'Cl': 1, 'As': 3,
                 'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class SampleAnalyzer():

    def __init__(self):
        pass

    def analyze_sample(self, sampled_molecules: List[SampledMolecule]):
        

        # compute the atom-level stabiltiy of a molecule. this is the number of atoms that have valid valencies.
        # note that since is computed at the atom level, even if the entire molecule is unstable, we can still get an idea
        # of how close the molecule is to being stable.
        n_atoms = 0
        n_stable_atoms = 0
        n_stable_molecules = 0
        n_molecules = len(sampled_molecules)
        for molecule in sampled_molecules:
            n_atoms += molecule.num_atoms
            n_stable_atoms_this_mol, mol_stable = check_stability(molecule)
            n_stable_atoms += n_stable_atoms_this_mol
            n_stable_molecules += int(mol_stable)

        frac_atoms_stable = n_stable_atoms / n_atoms # the fraction of generated atoms that have valid valencies
        frac_mols_stable_valence = n_stable_molecules / n_molecules # the fraction of generated molecules whose atoms all have valid valencies

        # TODO: compute metrics that come from MiDi (connecitivty, fragment size, etc.)


def check_stability(molecule: SampledMolecule):
    """ molecule: Molecule object. """
    atom_types = molecule.atom_types
    # edge_types = molecule.bond_types

    valencies = molecule.valencies

    n_stable_atoms = 0
    mol_stable = True
    for i, (atom_type, valency, charge) in enumerate(zip(atom_types, valencies, molecule.atom_charges)):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_type]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == valency
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[charge] if charge in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == valency if type(expected_bonds) == int else valency in expected_bonds
        else:
            is_stable = valency in possible_bonds
        if not is_stable:
            mol_stable = False
        n_stable_atoms += int(is_stable)

    return n_stable_atoms, mol_stable