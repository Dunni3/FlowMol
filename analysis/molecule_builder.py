import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
import dgl
from typing import List

bond_type_map = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]


class SampledMolecule:

    def __init__(self, g: dgl.DGLGraph, atom_type_map: List[str]):
        """Represents a molecule sampled from a model. Converts the DGL graph to an rdkit molecule and keeps all associated information."""

        # save the graph
        self.g = g

        self.atom_type_map = atom_type_map
        self.num_atoms = g.num_nodes()
        self.num_atom_types = len(atom_type_map)

        # extract node-level features
        self.positions = g.ndata['x_1']
        self.atom_types = g.ndata['a_1'].argmax(dim=1)
        self.atom_types = [atom_type_map[int(atom)] for atom in self.atom_types]
        self.atom_charges = g.ndata['c_1'].argmax(dim=1) - 2 # implicit assumption that index 0 charge is -2


        # get bond types and atom indicies for every edge, convert types from simplex to integer
        bond_types = g.edata['e_1'].argmax(dim=1)
        bond_src_idxs, bond_dst_idxs = g.edges()

        # get just the upper triangle of the adjacency matrix
        upper_edge_mask = g.edata['ue_mask']
        bond_types = bond_types[upper_edge_mask]
        bond_src_idxs = bond_src_idxs[upper_edge_mask]
        bond_dst_idxs = bond_dst_idxs[upper_edge_mask]

        # get only non-zero bond types
        bond_mask = bond_types != 0
        bond_types = bond_types[bond_mask]
        bond_src_idxs = bond_src_idxs[bond_mask]
        bond_dst_idxs = bond_dst_idxs[bond_mask]

        # save bond information
        self.bond_types = bond_types
        self.bond_src_idxs = bond_src_idxs
        self.bond_dst_idxs = bond_dst_idxs

        # build rdkit molecule
        self.rdkit_mol = self.build_molecule()

        # compute valencies on every atom
        self.valencies = self.compute_valencies()

    # this code is adapted from MiDi: https://github.com/cvignac/MiDi/blob/ba07fc5b1313855c047ba0b90e7aceae47e34e38/midi/analysis/rdkit_functions.py
    def build_molecule(self):
        # create a rdkit molecule and add atoms to it
        mol = Chem.RWMol()
        for atom_type, charge in zip(self.atom_types, self.atom_charges):
            a = Chem.Atom(atom_type)
            if charge != 0:
                a.SetFormalCharge(int(charge))
            mol.AddAtom(a)

        # add bonds to rdkit molecule
        for bond_type, src_idx, dst_idx in zip(self.bond_types, self.bond_src_idxs, self.bond_dst_idxs):
            src_idx = int(src_idx)
            dst_idx = int(dst_idx)
            mol.AddBond(src_idx, dst_idx, bond_type_map[bond_type])

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            return None

        # Set coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            x, y, z = self.positions[i]
            x, y, z = float(x), float(y), float(z)
            conf.SetAtomPosition(i, Point3D(x,y,z))
        mol.AddConformer(conf)

        return mol
    
    def compute_valencies(self):
        """Compute the valencies of every atom in the molecule. Returns a tensor of shape (num_atoms,)."""
        adj = torch.zeros((self.num_atoms, self.num_atoms))
        adjusted_bond_types = self.bond_types.clone()
        adjusted_bond_types[adjusted_bond_types == 4] = 1.5
        adj[self.bond_src_idxs, self.bond_dst_idxs] = adjusted_bond_types
        valencies = torch.sum(adj, dim=-1).long()
        return valencies