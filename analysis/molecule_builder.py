import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
import dgl
from typing import List

bond_type_map = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]

# this code is adapted from MiDi: https://github.com/cvignac/MiDi/blob/ba07fc5b1313855c047ba0b90e7aceae47e34e38/midi/analysis/rdkit_functions.py
def build_molecule(self, g: dgl.DGLGraph, atom_type_map: List[str]):


    # get positions, atom types, atom charges
    positions = g.ndata['x_1'].item()
    atom_types = g.ndata['a_1'].argmax(dim=1).item()
    atom_charges = g.ndata['c_1'].argmax(dim=1).item()

    # get bond types and atom indicies for every edge
    bond_types = g.edata['e_1']
    bond_src_idxs, bond_dst_idxs = g.edges()

    # get just the upper triangle of the adjacency matrix
    upper_edge_mask = g.edata['ue_mask']
    bond_types = bond_types[upper_edge_mask]
    bond_src_idxs = bond_src_idxs[upper_edge_mask]
    bond_dst_idxs = bond_dst_idxs[upper_edge_mask]

    # convert bond_types from simplex to integer
    bond_types = bond_types.argmax(dim=1)

    # get only non-zero bond types
    bond_mask = bond_types != 0
    bond_types = bond_types[bond_mask].item()
    bond_src_idxs = bond_src_idxs[bond_mask].item()
    bond_dst_idxs = bond_dst_idxs[bond_mask].item()


    # create a rdkit molecule and add atoms to it
    mol = Chem.RWMol()
    for atom, charge in zip(atom_types, atom_charges):
        if atom == -1:
            continue
        a = Chem.Atom(atom_type_map[int(atom)])
        if charge != 0:
            a.SetFormalCharge(charge)
        mol.AddAtom(a)

    # add bonds to rdkit molecule
    for bond_type, src_idx, dst_idx in zip(bond_types, bond_src_idxs, bond_dst_idxs):
        mol.AddBond(src_idx, dst_idx, bond_type_map[bond_type])

    try:
        mol = mol.GetMol()
    except Chem.KekulizeException:
        print("Can't kekulize molecule")
        return None

    # Set coordinates
    # note that the original code bothered to set positions.double() but I don't think that's necessary...just writing this incase I'm wrong
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(positions[i,0], positions[i,1], positions[i,2]))
    mol.AddConformer(conf)

    return mol