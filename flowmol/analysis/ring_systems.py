from useful_rdkit_utils.ring_systems import RingSystemLookup
from collections import defaultdict

class RingSystemCounter:

    def __init__(self):
        self.ring_system_lookup = RingSystemLookup.default()

    def count_ring_systems(self, rdmols):
        """
        Accepts a list of RDKit molecules and returns two dictioniaries: 
        one with the counts of ring systems observed in the sample, 
        and the other with frequencies of those ring systems in ChEMBL.
        """
        sample_counts = defaultdict(int)
        chembl_counts = {}
        for mol in rdmols:
            mol_ring_systems = self.ring_system_lookup.process_mol(mol)
            for ring_system_smi, chembl_count in mol_ring_systems:
                sample_counts[ring_system_smi] += 1
                chembl_counts[ring_system_smi] = chembl_count
        return sample_counts, chembl_counts
    
