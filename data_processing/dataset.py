import torch
from pathlib import Path
import dgl
from torch.nn.functional import one_hot
from data_processing.priors import compute_ot_prior, biased_simplex_prior, uniform_simplex_prior

# create a function named collate that takes a list of samples from the dataset and combines them into a batch
# this might not be necessary. I think we can pass the argument collate_fn=dgl.batch to the DataLoader
def collate(graphs):
    return dgl.batch(graphs)

class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, dataset_config: dict, prior_config: dict):
        super(MoleculeDataset, self).__init__()

        # unpack some configs regarding the prior
        self.prior_config = prior_config
        self.use_ot_node_feats: bool = prior_config['ot_node_feats']
        self.use_biased_edge_prior: bool = prior_config['biased_edge_prior']

        processed_data_dir: Path = Path(dataset_config['processed_data_dir'])

        if dataset_config['dataset_name'] == 'geom':
            data_file = processed_data_dir / f'{split}_data_processed.pt'
        else:
            raise NotImplementedError('unsupported dataset_name')

        # load data from processed data directory
        data_dict = torch.load(data_file)

        self.positions = data_dict['positions']
        self.atom_types = data_dict['atom_types']
        self.atom_charges = data_dict['atom_charges']
        self.bond_types = data_dict['bond_types']
        self.bond_idxs = data_dict['bond_idxs']
        self.node_idx_array = data_dict['node_idx_array']
        self.edge_idx_array = data_dict['edge_idx_array']

    def __len__(self):
        return self.node_idx_array.shape[0]
    
    def __getitem__(self, idx):
        node_start_idx = self.node_idx_array[idx, 0]
        node_end_idx = self.node_idx_array[idx, 1]
        edge_start_idx = self.edge_idx_array[idx, 0]
        edge_end_idx = self.edge_idx_array[idx, 1]
        
        # get data pertaining to nodes for this molecule
        positions = self.positions[node_start_idx:node_end_idx]
        atom_types = self.atom_types[node_start_idx:node_end_idx].float()
        atom_charges = self.atom_charges[node_start_idx:node_end_idx].long()

        # remove COM from positions
        positions = positions - positions.mean(dim=0, keepdim=True)

        # get data pertaining to edges for this molecule
        bond_types = self.bond_types[edge_start_idx:edge_end_idx].int()
        bond_idxs = self.bond_idxs[edge_start_idx:edge_end_idx].long()

        # reconstruct adjacency matrix
        n_atoms = positions.shape[0]
        adj = torch.zeros((n_atoms, n_atoms), dtype=torch.int32)

        # fill in the values of the adjacency matrix specified by bond_idxs
        adj[bond_idxs[:,0], bond_idxs[:,1]] = bond_types

        # get upper triangle of adjacency matrix
        upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1) # has shape (2, n_upper_edges)
        upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]

        # get lower triangle edges by swapping source and destination of upper_edge_idxs
        lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

        edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
        edge_labels = torch.cat((upper_edge_labels, upper_edge_labels))

        # one-hot encode edge labels and atom charges
        edge_labels = one_hot(edge_labels.to(torch.int64), num_classes=5).float() # hard-coded assumption of 5 bond types
        try:
            atom_charges = one_hot(atom_charges + 2, num_classes=6).float() # hard-coded assumption that charges are in range [-2, 3]
        except Exception as e:
            print('an atom charge outside of the expected range was encountered')
            print(f'max atom charge: {atom_charges.max()}, min atom charge: {atom_charges.min()}')
            raise e

        # create a dgl graph
        g = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)

        # add edge features
        g.edata['e_1_true'] = edge_labels

        # add node features
        g.ndata['x_1_true'] = positions
        g.ndata['a_1_true'] = atom_types
        g.ndata['c_1_true'] = atom_charges

        # sample prior with OT alignment on node features if necessary
        if self.use_ot_node_feats:
            dst_dict = {
                'x': positions,
                'a': atom_types,
                'c': atom_charges
            }

            prior_node_feats = compute_ot_prior(dst_dict, 
                                                pos_prior_std=self.prior_config['position_std'],
                                                rotate_positions=self.prior_config['rotate_positions']
                                                )

            for feat in prior_node_feats:
                g.ndata[f'{feat}_0'] = prior_node_feats[feat]

        # sample the prior for the edge features
        # note the hard-coded assumption of 5 bond types
        n_upper_edges = upper_edge_idxs.shape[1]
        if self.use_biased_edge_prior:
            edge_prior = biased_simplex_prior(n_upper_edges, 
                                              zero_order_weight=self.prior_config['no_bond_prob'], 
                                              std=self.prior_config['bond_order_std'], 
                                              d=5)
        else:
            edge_prior = uniform_simplex_prior(n_upper_edges, d=5)

        g.edata['e_0'] = torch.zeros_like(g.edata['e_1_true'])
        g.edata['e_0'][:n_upper_edges] = edge_prior # edge_prior has shape (n_upper_edges, 5)
        g.edata['e_0'][n_upper_edges:] = edge_prior

        return g