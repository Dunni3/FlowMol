from useful_rdkit_utils.ring_systems import RingSystemLookup
from collections import defaultdict
import numpy as np
import pandas as pd

from typing import Union, List, Dict, Tuple

class RingSystemCounter:

    def __init__(self):
        self.ring_system_lookup = RingSystemLookup.default()

    def count_ring_systems(self, rdmols: list) -> Tuple[Dict[str, int], Dict[str, int], int]:
        """
        Accepts a list of RDKit molecules and returns two dictioniaries: 
        one with the counts of ring systems observed in the sample, 
        and the other with frequencies of those ring systems in ChEMBL.
        """
        sample_counts = defaultdict(int)
        chembl_counts = {}
        n_mols = len(rdmols)
        for mol in rdmols:
            mol_ring_systems = self.ring_system_lookup.process_mol(mol)
            for ring_system_smi, chembl_count in mol_ring_systems:
                sample_counts[ring_system_smi] += 1
                chembl_counts[ring_system_smi] = chembl_count
        return sample_counts, chembl_counts, n_mols
    
    def combine_counts(self, counts_list: List[Tuple[Dict[str, int], Dict[str, int], int]]) -> Tuple[Dict[str, int], Dict[str, int], int]:
        """
        Accepts a list of tuples, each containing two dictionaries and an integer: 
        one with the counts of ring systems observed in the sample, 
        and the other with frequencies of those ring systems in ChEMBL.
        the integer is the number of molecules in the sample.
        Returns two dictionaries: one with the combined counts of ring systems observed in the sample, 
        and the other with frequencies of those ring systems in ChEMBL. Also the total number of molecules in the sample.
        """
        combined_sample_counts = defaultdict(int)
        combined_chembl_counts = {}
        n_mols_combined = 0
        for sample_counts, chembl_counts, n_mols in counts_list:
            n_mols_combined += n_mols
            for ring_system_smi, count in sample_counts.items():
                combined_sample_counts[ring_system_smi] += count
            for ring_system_smi, count in chembl_counts.items():
                combined_chembl_counts[ring_system_smi] = count
        return combined_sample_counts, combined_chembl_counts, n_mols_combined
    
def ring_counts_to_df(sample_counts, chembl_counts, n_mols) -> pd.DataFrame:

    # create df of ring rates
    ring_smi = list(sample_counts.keys())
    ring_counts = np.array([sample_counts[smi] for smi in ring_smi])
    chembl_counts = np.array([chembl_counts[smi] for smi in ring_smi])
    sample_rate = ring_counts / n_mols

    df_rings = pd.DataFrame({
        'ring_smi': ring_smi,
        'sample_rate': sample_rate,
        'sample_count': ring_counts,
        'n_mols': n_mols,
        'chembl_count': chembl_counts
    })
    return df_rings