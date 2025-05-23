import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
import dgl
from typing import List, Dict
from flowmol.data_processing.priors import rigid_alignment
from flowmol.data_processing.utils import get_upper_edge_mask
from torch.nn.functional import one_hot

bond_type_map = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC, None] # last bond type is for masked bonds

bond_type_to_idx = { bond_type:idx for idx, bond_type in enumerate(bond_type_map)}
bond_type_to_idx[None] = 0


class SampledMolecule:

    def __init__(self, g: dgl.DGLGraph, 
        atom_type_map: List[str], 
        traj_frames: Dict[str, torch.Tensor] = None,
        ctmc_mol: bool = False, # whether the molecule was sampled from a CTMC model. Important because one-hot encodings will contain a mask token. 
        fake_atoms: bool = False, # whether the molecule contains fake atoms,
        exclude_charges: bool = False, 
        align_traj: bool = True,
        build_xt_traj=True,
        build_ep_traj=True,
        explicit_aromaticity: bool = False,
    ):
        """Represents a molecule sampled from a model. Converts the DGL graph to an rdkit molecule and keeps all associated information."""

        atom_type_map = list(atom_type_map) # create a shallow copy of the atom type map so that we don't modify the original

        self.exclude_charges = exclude_charges
        self.align_traj = align_traj
        self.ctmc_mol = ctmc_mol
        self.fake_atoms = fake_atoms
        self.explicit_aromaticity = explicit_aromaticity

        if fake_atoms:
            atom_type_map.append('Sn') # fake atoms will show up as tin, only in trajectories

        if ctmc_mol:
            atom_type_map.append('Se') # masked molecules will show up as selenium
        
        # save the graph
        self.g = g

        self.positions, self.atom_types, self.atom_charges, self.bond_types, self.bond_src_idxs, self.bond_dst_idxs = extract_moldata_from_graph(
            copy_graph(g), 
            atom_type_map, 
            exclude_charges=exclude_charges,
            ctmc_mol=self.ctmc_mol,
            fake_atoms=self.fake_atoms,
            show_fake_atoms=False)

        self.atom_type_map = atom_type_map
        self.num_atom_types = len(atom_type_map)

        self.num_atoms = g.num_nodes()
        if self.fake_atoms:
            fake_atom_token_idx  = len(atom_type_map) - 2 
            fake_atom_mask = g.ndata['a_1'].argmax(dim=1) == fake_atom_token_idx
            n_fake_atoms = fake_atom_mask.sum().item()
            self.num_atoms -= n_fake_atoms

        # build rdkit molecule
        self.rdkit_mol = self.build_molecule()

        # compute valencies on every atom
        self.valencies = self.compute_valencies(arom_dependent=explicit_aromaticity)

        # build trajectory molecules
        self.traj_frames = traj_frames
        if traj_frames is not None:

            if build_xt_traj:
                self.traj_mols = self.process_traj_frames(traj_frames) # convert frames into a list of rdkit molecules

            # construct molecules for endpoint trajectory
            if 'x_1_pred' in traj_frames and build_ep_traj:
                self.ep_traj_mols = self.process_traj_frames(traj_frames, ep_traj=True)

    @classmethod
    def from_rdkit_mol(cls, mol: Chem.Mol, atom_type_map: List[str] = None):
        """Creates a SampledMolecule from an rdkit molecule."""

        if atom_type_map is None:
            atom_types = [ atom.GetSymbol() for atom in mol.GetAtoms() ]
            atom_type_map = list(set(atom_types))

        atom_type_to_idx = {atom_type: i for i, atom_type in enumerate(atom_type_map)}

        # get positions
        positions = mol.GetConformer().GetPositions()
        positions = torch.from_numpy(positions)

        # get atom elements as a string
        # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        atom_types_idx = torch.zeros(mol.GetNumAtoms()).long()
        atom_charges = torch.zeros_like(atom_types_idx)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_types_idx[i] = atom_type_to_idx[atom.GetSymbol()]
            atom_charges[i] = atom.GetFormalCharge()

        # get one-hot encoded of existing bonds only (no non-existing bonds)
        adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
        edge_index = adj.triu().nonzero().contiguous() # upper triangular portion of adjacency matrix

        bond_types = adj[edge_index[:, 0], edge_index[:, 1]]
        bond_src_idxs = edge_index[:, 0]
        bond_dst_idxs = edge_index[:, 1]
        bond_types[bond_types == 1.5] = 4
        edge_attr = bond_types.type(torch.int32)

        # create graph
        g = dgl.graph((bond_src_idxs, bond_dst_idxs), num_nodes=mol.GetNumAtoms())

        # add data to the graph
        g.ndata['x_1'] = positions
        g.ndata['a_1'] = one_hot(atom_types_idx.long(), num_classes=len(atom_type_map)).float()
        g.ndata['c_1'] = one_hot(atom_charges.long() + 2, num_classes=6).float()
        g.edata['e_1'] = one_hot(edge_attr.long(), num_classes=5).float()
        g.edata['ue_mask'] = torch.ones(g.num_edges()).bool()

        return cls(g, atom_type_map=atom_type_map)
    
    # this code is adapted from MiDi: https://github.com/cvignac/MiDi/blob/ba07fc5b1313855c047ba0b90e7aceae47e34e38/midi/analysis/rdkit_functions.py
    def build_molecule(self):
        mol = build_molecule(self.positions, self.atom_types, self.atom_charges, self.bond_src_idxs, self.bond_dst_idxs, self.bond_types)
        return mol
    
    def compute_valencies(self, arom_dependent: bool = False):
        """Compute the valencies of every atom in the molecule. Returns a tensor of shape (num_atoms,)."""

        if arom_dependent:
            raise NotImplementedError("Aromaticity dependent valency computation is not implemented yet.")

        adj = torch.zeros((self.num_atoms, self.num_atoms)).float()
        adjusted_bond_types = self.bond_types.clone().float().float()
        adjusted_bond_types[adjusted_bond_types == 4] = 1.5
        adjusted_bond_types = adjusted_bond_types.float()
        adj[self.bond_src_idxs, self.bond_dst_idxs] = adjusted_bond_types
        adj[self.bond_dst_idxs, self.bond_src_idxs] = adjusted_bond_types


        valencies = torch.sum(adj, dim=-1)

        if arom_dependent:
            n_arom = (adj == 1.5).sum(dim=-1)
            non_arom_valence = (valencies - n_arom*1.5).long()
            valencies = torch.stack([n_arom, non_arom_valence], dim=1)

        
        return valencies
    
    def process_traj_frames(self, traj_frames: Dict[str, torch.Tensor], ep_traj: bool = False):
        """Converts the trajectory frames to a list of rdkit molecules."""
        # convert the frames to a list of rdkit molecules
        g_dummy = copy_graph(self.g)

        if ep_traj:
            n_frames = traj_frames['x_1_pred'].shape[0]
            x_final = traj_frames['x_1_pred'][-1]
        else:
            n_frames = traj_frames['x'].shape[0]
            x_final = traj_frames['x'][-1] # has shape (n_atoms, 3)

        traj_mols = []
        for frame_idx in range(n_frames):

            # put current frame data into graph
            for feat in traj_frames.keys():

                if '1_pred' in feat:
                    continue

                if feat == 'e':
                    data_src = g_dummy.edata
                else:
                    data_src = g_dummy.ndata

                if ep_traj:
                    traj_key = f'{feat}_1_pred'
                else:
                    traj_key = feat

                data_src[f'{feat}_1'] = traj_frames[traj_key][frame_idx].clone()

            # extract mol data from graph
            positions, atom_types, atom_charges, bond_types, bond_src_idxs, bond_dst_idxs = extract_moldata_from_graph(
                g_dummy, 
                self.atom_type_map,
                ctmc_mol=self.ctmc_mol,
                fake_atoms=self.fake_atoms,
                show_fake_atoms=True, # for trajectories, we want to show fake atoms
                )

            # align positions to final frame
            if self.align_traj:
                positions = rigid_alignment(positions, x_final)

            # build rdkit molecule
            mol = build_molecule(positions, atom_types, atom_charges, bond_src_idxs, bond_dst_idxs, bond_types)

            # add mol to list
            traj_mols.append(mol)

        traj_mols = [mol for mol in traj_mols if mol is not None]

        if len(traj_mols) < len(traj_frames):
            print(f'WARNING: {len(traj_frames) - len(traj_mols)} frames were not converted to rdkit molecules')

        return traj_mols
    

def extract_moldata_from_graph(g: dgl.DGLGraph, atom_type_map: List[str], exclude_charges: bool = False, ctmc_mol: bool = False, fake_atoms: bool = False, show_fake_atoms: bool = False):

    # if fake atoms are present, identify them
    if fake_atoms and not show_fake_atoms:
        fake_atom_token_idx  = len(atom_type_map) - 2 
        fake_atom_mask = g.ndata['a_1'].argmax(dim=1) == fake_atom_token_idx
        fake_atom_idxs = torch.where(fake_atom_mask)[0]
        g.remove_nodes(fake_atom_idxs)

    # extract node-level features
    positions = g.ndata['x_1']

    # extract node-level features
    positions = g.ndata['x_1']
    atom_types = g.ndata['a_1'].argmax(dim=1)
    atom_types = [atom_type_map[int(atom)] for atom in atom_types]

    if exclude_charges:
        atom_charges = None
    else:
        atom_charges = g.ndata['c_1'].argmax(dim=1) - 2 # implicit assumption that index 0 charge is -2

    # get bond types and atom indicies for every edge, convert types from simplex to integer
    bond_types = g.edata['e_1'].argmax(dim=1)
    bond_types[bond_types == 4] = 0 # set masked bonds to 0
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

    return positions, atom_types, atom_charges, bond_types, bond_src_idxs, bond_dst_idxs


def build_molecule(positions, atom_types, atom_charges, bond_src_idxs, bond_dst_idxs, bond_types):
    """Builds a rdkit molecule from the given atom and bond information."""
    # create a rdkit molecule and add atoms to it
    mol = Chem.RWMol()
    for atom_type, charge in zip(atom_types, atom_charges):
        a = Chem.Atom(atom_type)
        if charge != 0:
            a.SetFormalCharge(int(charge))
        mol.AddAtom(a)

    # add bonds to rdkit molecule
    for bond_type, src_idx, dst_idx in zip(bond_types, bond_src_idxs, bond_dst_idxs):
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
        x, y, z = positions[i]
        x, y, z = float(x), float(y), float(z)
        conf.SetAtomPosition(i, Point3D(x,y,z))
    mol.AddConformer(conf)

    return mol


def copy_graph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    
    # get edges
    edges = g.edges(form='uv')

    # get number of nodes
    num_nodes = g.num_nodes()

    g_copy = dgl.graph(edges, num_nodes=num_nodes, device=g.device)

    # transfer over node features
    for nfeat in g.ndata.keys():
        g_copy.ndata[nfeat] = g.ndata[nfeat].detach().clone()

    # transfer over edge features
    for efeat in g.edata.keys():
        g_copy.edata[efeat] = g.edata[efeat].detach().clone()


    return g_copy

def dataset_mol_to_sampled_mol(g, atom_type_map) -> SampledMolecule:
    for feat in 'xace':
        if feat == 'e':
            data_src = g.edata
        else:
            data_src = g.ndata
        data_src[f'{feat}_1'] = data_src[f'{feat}_1_true']

    g.edata['ue_mask'] = get_upper_edge_mask(g)
    return SampledMolecule(g, atom_type_map)

def dataset_mol_to_rdmol(g, atom_type_map):
    dataset_mol_to_sampled_mol(g, atom_type_map).rdkit_mol