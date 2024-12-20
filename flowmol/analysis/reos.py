import useful_rdkit_utils.reos as reos
import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from typing import List

class REOS:
    """Wrapper class for userful_rdkit_utils.reos.REOS that implements structural alert counting in a way that suits our needs."""

    def __init__(self, *args, **kwargs):
        self.reos = reos.REOS(*args, **kwargs)


        # collect all rule sets into an ordered list of rule names
        self.flag_arr_header = []
        self.smarts_arr = []
        for desc, rule_set_name, smarts in self.reos.active_rule_df[['description', 'rule_set_name', 'smarts']].values:
            self.flag_arr_header.append(f"{rule_set_name}::{desc}")
            self.smarts_arr.append(smarts)

        # argsort flag_arr_header and reorder flag_arr_header and smarts_arr
        argsort_indices = np.argsort(self.flag_arr_header)
        self.flag_arr_header = [self.flag_arr_header[i] for i in argsort_indices]
        self.smarts_arr = [self.smarts_arr[i] for i in argsort_indices]

    def mol_to_flags(self, mol):
        """Match a molecule against the active rule set

        :param mol: input RDKit molecule
        :return: a set containing the names of all rules that matched
        """
        cols = ['description', 'rule_set_name', 'smarts', 'pat', 'max']


        flags_found = set()
        for desc, rule_set_name, smarts, pat, max_val in self.reos.active_rule_df[cols].values:
            if len(mol.GetSubstructMatches(pat)) > max_val:
                matched = True
            else:
                matched = False

            if matched:
                flags_found.add(f"{rule_set_name}::{desc}")
                

        return flags_found
    
    def mols_to_flag_arr(self, mol_list: List[Mol]):

        rows = []
        for mol in mol_list:
            row = np.zeros(len(self.flag_arr_header), dtype=bool)

            for flag_name in self.mol_to_flags(mol):
                row[self.flag_arr_header.index(flag_name)] = 1


            rows.append(row)

        return np.stack(rows)