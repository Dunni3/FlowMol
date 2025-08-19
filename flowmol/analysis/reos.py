import useful_rdkit_utils.reos as reos
import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from typing import List
import pandas as pd

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
    

def build_reos_df(flag_arr, flag_names):
    flag_rates = flag_arr.sum(0) / flag_arr.shape[0]
    df_flags = pd.DataFrame({
        'flag_name': flag_names,
        'flag_count': flag_arr.sum(0),
        'flag_rate': flag_rates,
        'n_mols': flag_arr.shape[0],
    })

    avg_flag_rate = flag_arr.sum() / flag_arr.shape[0]

    # data_frequency_map = { flag_name: flag_rate for flag_name, flag_rate in zip(flag_names, flag_rates) }

    # has one flag rate
    has_flags_rate = (flag_arr.sum(1) > 0).sum() / flag_arr.shape[0]

    reos_metrics = {
        'avg_flag_rate': avg_flag_rate,
        'has_flags_rate': has_flags_rate,
    }

    return df_flags