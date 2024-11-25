import torch
from pathlib import Path
import dgl
from torch.nn.functional import one_hot
from flowmol.data_processing.priors import coupled_node_prior, edge_prior

# create a function named collate that takes a list of samples from the dataset and combines them into a batch
# this might not be necessary. I think we can pass the argument collate_fn=dgl.batch to the DataLoader
def collate(graphs):
    return dgl.batch(graphs)

class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, split: str, dataset_config: dict, prior_config: dict):
        super(MoleculeDataset, self).__init__()

        # unpack some configs regarding the prior
        self.prior_config = prior_config
        self.dataset_config = dataset_config

        # get the processed data directory
        processed_data_dir: Path = Path(dataset_config['processed_data_dir'])

        # if the processed data directory does not exist, check it relative to the root of flowmol repository
        if not processed_data_dir.exists():
            processed_data_dir = Path(__file__).parent.parent.parent / processed_data_dir
            if processed_data_dir.exists():
                dataset_config['processed_data_dir'] = str(processed_data_dir)
            else:
                raise FileNotFoundError(f"processed data directory {dataset_config['processed_data_dir']} not found.")
            
        self.processed_data_dir = processed_data_dir

        # load the marginal distributions of atom types and the conditional distribution of charges given atom type
        marginal_dists_file = processed_data_dir / 'train_data_marginal_dists.pt'
        p_a, p_c, p_e, p_c_given_a = torch.load(marginal_dists_file)

        # add the marginal distributions as arguments to the prior sampling functions
        if self.prior_config['a']['type'] == 'marginal':
            self.prior_config['a']['kwargs']['p'] = p_a

        if self.prior_config['e']['type'] == 'marginal':
            self.prior_config['e']['kwargs']['p'] = p_e

        if self.prior_config['c']['type'] == 'marginal':
            self.prior_config['c']['kwargs']['p'] = p_c
        
        if self.prior_config['c']['type'] == 'c-given-a':
            self.prior_config['c']['kwargs']['p_c_given_a'] = p_c_given_a

        if dataset_config['dataset_name'] in ['geom', 'qm9', 'geom_5conf']:
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

        # sample prior for node features, coupled to the destination features
        dst_dict = {
            'x': positions,
            'a': atom_types,
            'c': atom_charges
        }
        prior_node_feats = coupled_node_prior(dst_dict=dst_dict, prior_config=self.prior_config)
        for feat in prior_node_feats:
            g.ndata[f'{feat}_0'] = prior_node_feats[feat]

        # sample the prior for the edge features    
        upper_edge_mask = torch.zeros(g.num_edges(), dtype=torch.bool)
        n_upper_edges = upper_edge_idxs.shape[1]
        upper_edge_mask[:n_upper_edges] = True
        g.edata['e_0'] = edge_prior(upper_edge_mask, self.prior_config['e'])

        return g