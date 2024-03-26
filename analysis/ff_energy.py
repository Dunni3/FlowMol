from rdkit.Chem import AllChem as Chem
from typing import List

def compute_uff_energy(mol):
    ff = Chem.UFFGetMoleculeForceField(mol, ignoreInterfragInteractions=False)
    return ff.CalcEnergy()

def compute_mmff_energy(mol):
    try:
        ff = Chem.MMFFGetMoleculeForceField(mol, Chem.MMFFGetMoleculeProperties(mol), ignoreInterfragInteractions=False)
    except Exception as e:
        print(e)
        print('Failed to get force-field object')
        return None
    if ff:
        return ff.CalcEnergy()
    return None


