from typing import List
from .molecule_builder import SampledMolecule
from pathlib import Path
import torch
from rdkit import Chem
from collections import Counter
import wandb
from flowmol.utils.divergences import DivergenceCalculator
from flowmol.analysis.ff_energy import compute_mmff_energy
from flowmol.analysis.reos import REOS
from flowmol.analysis.ring_systems import RingSystemCounter, ring_counts_to_df

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


class SampleAnalyzer():

    def __init__(self, processed_data_dir: str = None, dataset='geom'):

        self.processed_data_dir = processed_data_dir

        if self.processed_data_dir is None:
            self.processed_data_dir = Path(__file__).parent.parent.parent / 'data' / dataset

        energy_dist_file = self.processed_data_dir / 'energy_dist.npz'
        self.energy_div_calculator = DivergenceCalculator(energy_dist_file)
            

    def analyze(self, sampled_molecules: List[SampledMolecule], return_counts: bool = False, energy_div: bool = False, functional_validity: bool = False):

        # compute the atom-level stabiltiy of a molecule. this is the number of atoms that have valid valencies.
        # note that since is computed at the atom level, even if the entire molecule is unstable, we can still get an idea
        # of how close the molecule is to being stable.
        n_atoms = 0
        n_stable_atoms = 0
        n_stable_molecules = 0
        n_molecules = len(sampled_molecules)
        for molecule in sampled_molecules:
            n_stable_atoms_this_mol, mol_stable, n_fake_atoms = check_stability(molecule)
            n_atoms += molecule.num_atoms - n_fake_atoms
            n_stable_atoms += n_stable_atoms_this_mol
            n_stable_molecules += int(mol_stable)

        frac_atoms_stable = n_stable_atoms / n_atoms # the fraction of generated atoms that have valid valencies
        frac_mols_stable_valence = n_stable_molecules / n_molecules # the fraction of generated molecules whose atoms all have valid valencies

        # compute validity as determined by rdkit, and the average size of the largest fragment, and the average number of fragments
        validity_result = self.compute_validity(sampled_molecules, return_counts=return_counts)
        if return_counts:
            frac_valid_mols, avg_frag_frac, avg_num_components, n_valid, sum_frag_fracs, n_frag_fracs, sum_num_components, n_num_components = validity_result
        else:
            frac_valid_mols, avg_frag_frac, avg_num_components = validity_result

        metrics_dict = {
            'frac_atoms_stable': frac_atoms_stable,
            'frac_mols_stable_valence': frac_mols_stable_valence,
            'frac_valid_mols': frac_valid_mols,
            'avg_frag_frac': avg_frag_frac,
            'avg_num_components': avg_num_components
        }
        # TODO: i think the return_counts functionality was so that we could
        # compute metrics on the  entire dataset by chunking it and combining the counts at the end
        # this functionality is not supported yet for reos and rings`
        # before we compute dataset-level metrics, we need to implement the functionality to combine counts
        # of reos/rings outputs. 
        if functional_validity:
            metrics_dict.update(self.reos_and_rings(sampled_molecules, return_raw=False))

        if return_counts:
            counts_dict = {}
            counts_dict['n_stable_atoms'] = n_stable_atoms
            counts_dict['n_atoms'] = n_atoms
            counts_dict['n_stable_molecules'] = n_stable_molecules
            counts_dict['n_molecules'] = n_molecules
            counts_dict['n_valid'] = n_valid
            counts_dict['sum_frag_fracs'] = sum_frag_fracs
            counts_dict['n_frag_fracs'] = n_frag_fracs
            counts_dict['sum_num_components'] = sum_num_components
            counts_dict['n_num_components'] = n_num_components
            return counts_dict
        
        if self.processed_data_dir is not None and Path(self.processed_data_dir).exists() and energy_div:
            metrics_dict['energy_js_div'] = self.compute_energy_divergence(sampled_molecules)


        return metrics_dict

    # this function taken from MiDi molecular_metrics.py script
    def compute_validity(self, sampled_molecules: List[SampledMolecule], return_counts: bool = False):
        """ generated: list of couples (positions, atom_types)"""
        n_valid = 0
        num_components = []
        frag_fracs = []
        error_message = Counter()
        for mol in sampled_molecules:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                    num_components.append(len(mol_frags))
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    largest_mol_n_atoms = largest_mol.GetNumAtoms()
                    largest_frag_frac = largest_mol_n_atoms / mol.num_atoms
                    frag_fracs.append(largest_frag_frac)
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol)
                    n_valid += 1
                    error_message[-1] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    # print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    # print("Can't kekulize molecule")
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
        print(f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
              f" -- No error {error_message[-1]}")
        

        frac_valid_mols = n_valid / len(sampled_molecules)
        avg_frag_frac = sum(frag_fracs) / len(frag_fracs)
        avg_num_components = sum(num_components) / len(num_components)

        if return_counts:
            return frac_valid_mols, avg_frag_frac, avg_num_components, n_valid, sum(frag_fracs), len(frag_fracs), sum(num_components), len(num_components)

        return frac_valid_mols, avg_frag_frac, avg_num_components

    def compute_sample_energy(self, samples: List[SampledMolecule]):
        """ samples: list of SampledMolecule objects. """
        energies = []
        for sample in samples:
            rdmol = sample.rdkit_mol
            if rdmol is not None:
                try:
                    Chem.SanitizeMol(rdmol)
                except:
                    continue
                energy = compute_mmff_energy(rdmol)
                if energy is not None:
                    energies.append(energy)

        return energies

    def compute_energy_divergence(self, samples: List[SampledMolecule]):

        if self.processed_data_dir is None:
            raise ValueError('You must specify processed_data_dir upon initialization to compute energy divergences')

        # compute the FF energy of each molecule
        energies = self.compute_sample_energy(samples)

        # compute the Jensen-Shannon divergence between the energy distribution of the samples and the training set
        js_div = self.energy_div_calculator.js_divergence(energies)

        return js_div

    def reos_and_rings(self, samples: List[SampledMolecule], return_raw=False):
        """ samples: list of SampledMolecule objects. """
        rd_mols = [sample.rdkit_mol for sample in samples]
        valid_idxs = []
        sanitized_mols = []
        for i, mol in enumerate(rd_mols):
            try:
                Chem.SanitizeMol(mol)
                sanitized_mols.append(mol)
                valid_idxs.append(i)
            except:
                continue
        reos = REOS(active_rules=["Glaxo", "Dundee"])
        ring_system_counter = RingSystemCounter()

        if len(sanitized_mols) != 0:
            reos_flags = reos.mols_to_flag_arr(sanitized_mols)
            ring_counts = ring_system_counter.count_ring_systems(sanitized_mols)
        else:
            reos_flags = None
            ring_counts = None

        if return_raw:
            result = {
                        'reos_flag_arr': reos_flags,
                        'reos_flag_header': reos.flag_arr_header,
                        'smarts_arr': reos.smarts_arr,
                        'ring_counts': ring_counts,
                        'valid_idxs': valid_idxs
                    }
            return result
        
        if reos_flags is not None:
            n_flags = reos_flags.sum()
            n_mols = reos_flags.shape[0]
            flag_rate = n_flags / n_mols

            sample_counts, chembl_counts, n_mols = ring_counts
            df_ring = ring_counts_to_df(sample_counts, chembl_counts, n_mols)
            ood_ring_count = df_ring[df_ring['chembl_count'] == 0]['sample_count'].sum()
            ood_rate = ood_ring_count / n_mols
        else:
            flag_rate = -1
            ood_rate = -1
        
        return dict(flag_rate=flag_rate, ood_rate=ood_rate)

def check_stability(molecule: SampledMolecule):
    """ molecule: Molecule object. """
    atom_types = molecule.atom_types
    valencies = molecule.valencies
    charges = molecule.atom_charges

    n_stable_atoms = 0
    n_fake_atoms = 0 
    mol_stable = True
    for i, (atom_type, valency, charge) in enumerate(zip(atom_types, valencies, charges)):

        if molecule.fake_atoms and atom_type == 'Sn':
            n_fake_atoms += 1
            continue

        valency = int(valency)
        charge = int(charge)
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

    return n_stable_atoms, mol_stable, n_fake_atoms