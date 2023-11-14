from typing import Dict
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import dgl
import torch.nn.functional as fn
from torch.distributions import Exponential

from .lr_scheduler import LRScheduler
from .interpolant_scheduler import InterpolantScheduler
from .vector_field import GVPVectorField

class MolFM(pl.LightningModule):

    canonical_feat_order = ['x', 'a', 'c', 'e']

    def __init__(self,
                 n_atom_types: int,                  
                 batches_per_epoch: int,
                 n_atoms_hist_file: str,
                 n_atom_charges: int = 6,
                 n_bond_types: int = 5,
                 time_scaled_loss: bool = True,
                 total_loss_weights: Dict[str, float] = {}, 
                 lr_scheduler_config: dict = {},
                 interpolant_scheduler_config: dict = {},
                 vector_field_config: dict = {}
                 ):
        super().__init__()

        self.batches_per_epoch = batches_per_epoch
        self.lr_scheduler_config = lr_scheduler_config
        self.n_atom_types = n_atom_types
        self.n_atom_charges = n_atom_charges
        self.n_bond_types = n_bond_types
        self.total_loss_weights = total_loss_weights
        self.time_scaled_loss = time_scaled_loss

        for feat in self.canonical_feat_order:
            if feat not in total_loss_weights:
                self.total_loss_weights[feat] = 1.0

                # print warning if the user has not specified a loss weight for a feature
                print(f'WARNING: no loss weight specified for feature {feat}, using default of 1.0')

        self.exp_dist = Exponential(1.0)
        
        # construct histogram of number of atoms in each ligand
        self.n_atoms_hist_file = n_atoms_hist_file
        self.build_n_atoms_dist(n_atoms_hist_file=n_atoms_hist_file)

        # create interpolant scheduler and vector field
        self.interpolant_scheduler = InterpolantScheduler(canonical_feat_order=self.canonical_feat_order, **interpolant_scheduler_config)
        self.vector_field = GVPVectorField(**vector_field_config)

        # loss functions
        if self.time_scaled_loss:
            reduction = 'none'
        else:
            reduction = 'mean'
        self.loss_fn_dict = {
            'x': nn.MSELoss(reduction=reduction),
            'a': nn.CrossEntropyLoss(reduction=reduction),
            'c': nn.CrossEntropyLoss(reduction=reduction),
            'e': nn.CrossEntropyLoss(reduction=reduction),
        }

        self.save_hyperparameters()

    def training_step(self, g: dgl.DGLGraph, batch_idx: int):

        epoch_exact = self.current_epoch + batch_idx/self.batches_per_epoch
        self.lr_scheduler.step_lr(epoch_exact)
        losses = self(g)

        self.log('epoch_exact', epoch_exact, on_step=True, prog_bar=True)

        for key in losses:
            self.log(f'{key}_loss', losses[key], on_step=True, prog_bar=True)

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]

        return total_loss
    
    def forward(self, g: dgl.DGLGraph):
        
        batch_size = g.batch_size
        device = g.device

        # get batch indicies of every atom and edge
        node_batch_idx = torch.arange(batch_size, device=device)
        node_batch_idx = node_batch_idx.repeat_interleave(g.batch_num_nodes())
        edge_batch_idx = torch.arange(batch_size, device=device)
        edge_batch_idx = edge_batch_idx.repeat_interleave(g.batch_num_edges())

        # create a mask which selects all of the upper triangle edges from the batched graph
        edges_per_mol = g.batch_num_edges()
        ul_pattern = torch.tensor([1,0]).repeat(batch_size)
        n_edges_pattern = (edges_per_mol/2).int().repeat_interleave(2)
        upper_edge_mask = ul_pattern.repeat_interleave(n_edges_pattern).bool()

        # get initial COM of each molecule and remove the COM from the atom positions
        init_coms = dgl.readout_nodes(g, feat='x_1_true', op='mean')
        g.ndata['x_1_true'] = g.ndata['x_1_true'] - init_coms[node_batch_idx]

        # sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        # sample timepoints for each molecule in the batch
        t = torch.rand(batch_size, device=device).float()

        # construct interpolated molecules
        g = self.interpolate(g, t, node_batch_idx, edge_batch_idx)

        # predict the end of the trajectory
        dst_dict = self.vector_field.pred_dst(g, t)

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)

        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            # get the target for this feature
            if feat == 'e':
                target = g.edata[f'{feat}_1_true']
            else:
                target = g.ndata[f'{feat}_1_true']

            # get the target as class indicies for categorical features
            if feat in ['a', 'c', 'e']:
                target = target.argmax(dim=-1)

            if self.time_scaled_loss:
                weight = time_weights.view(-1, feat_idx)
                if feat == 'e':
                    weight = weight[edge_batch_idx]
                else:
                    weight = weight[node_batch_idx]
            else:
                weight = 1.0

            # compute the losses
            losses[feat] = self.loss_fn_dict[feat](dst_dict[feat], target)*weight

            # when time_scaled_loss is True, we set the reduction to 'none' so that each loss can be scaled by the time-dependent weight.
            # however, this means that we also need to do the reduction ourselves here.
            if self.time_scaled_loss:
                losses[feat] = losses[feat].mean()

        return losses
    
    def sample_prior(self, g, node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor):
        """Sample from the prior distribution of the ligand."""
        # sample atom positions from prior
        # TODO: we should set the standard deviation of atom position prior to be like the average distance to the COM in the training set
        # or perhaps the average distance to COM for molecules with the same number of atoms
        # TODO: can we implement OT flow matching for the prior? equivariant flow-matching?
        g.ndata['x_0'] = torch.randn_like(g.ndata['x_1_true'])
        g.ndata['x_0'] = g.ndata['x_0'] - dgl.readout_nodes(g, feat='x_0', op='mean')[node_batch_idx]

        # sample atom types, charges from simplex
        # TODO: can we implement OT flow matching for the simplex prior?
        for node_feat in ['a', 'c']:
            g.ndata[f'{node_feat}_0'] = self.exp_dist.sample(g.ndata[f'{node_feat}_1_true'].shape)
            g.ndata[f'{node_feat}_0'] = g.ndata[f'{node_feat}_0'] / g.ndata[f'{node_feat}_0'].sum(dim=1, keepdim=True)

        # sample bond types from simplex prior - make sure the sample for the lower triangle is the same as the upper triangle
        n_upper_edges = upper_edge_mask.sum()
        g.edata['e_0'] = torch.zeros_like(g.edata['e_1_true'])
        e_0_upper = self.exp_dist.sample((n_upper_edges, self.n_bond_types))
        e_0_upper = e_0_upper / e_0_upper.sum(dim=1, keepdim=True)
        g.edata['e_0'][upper_edge_mask] = e_0_upper
        g.edata['e_0'][~upper_edge_mask] = e_0_upper

        return g
    
    def interpolate(self, g, t, node_batch_idx, edge_batch_idx):
        """Interpolate between the prior and true terminal state of the ligand."""
        # TODO: this computation could be made more efficient by concatenating node features and edge features into a single tensor and then interpolate them all at once before splitting them back up
        src_weights, dst_weights = self.interpolant_scheduler.interpolant_weights(t)

        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if feat == 'e':
                continue

            src_weight, dst_weight = src_weights[feat_idx][node_batch_idx], dst_weights[feat_idx][node_batch_idx]
            g.ndata[f'{feat}_t'] = src_weight * g.ndata[f'{feat}_0'] + dst_weight * g.ndata[f'{feat}_1_true']

        e_idx = self.canonical_feat_order.index('e')

        src_weight, dst_weight = src_weights[e_idx][edge_batch_idx], dst_weights[e_idx][edge_batch_idx]
        g.edata[f'e_t'] = src_weight * g.edata[f'e_0'] + dst_weight * g.edata[f'e_1_true']

        return g

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.scheduler_config['base_lr'])
        self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.scheduler_config)
        return optimizer

    def build_n_atoms_dist(self, n_atoms_hist_file: str):
        """Builds the distribution of the number of atoms in a ligand."""
        n_atoms, n_atom_counts = torch.load(n_atoms_hist_file)
        n_atoms_prob = n_atom_counts / n_atom_counts.sum()
        self.n_atoms_dist = torch.distributions.Categorical(probs=n_atoms_prob)
        self.n_atoms_map = n_atoms

    def sample_n_atoms(self, n_molecules: int):
        """Draw samples from the distribution of the number of atoms in a ligand."""
        n_atoms = self.n_atoms_dist.sample((n_molecules,))
        return self.n_atoms_map[n_atoms]

    def sample_random_sizes(self, n_molecules: int, device="cuda:0"):
        """Sample n_moceules with the number of atoms sampled from the distribution of the training set."""

        # get the number of atoms that will be in each molecules
        atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)

        return self.sample(atoms_per_molecule)

    def sample(self, n_atoms: torch.Tensor):
        """Sample molecules with the given number of atoms.
        
        Args:
            n_atoms (torch.Tensor): Tensor of number of atoms in each molecule.
        """

        # create a graph with the correct number of atoms
        g = dgl.batch([ dgl.graph(([], []), num_nodes=n) for n in n_atoms ])

        # sample random positions and features
        g.ndata['x_0'] = torch.randn((g.num_nodes(), 3))
        g.ndata['h_0'] = torch.randn((g.num_nodes(), self.n_atom_features))

        batch_size = g.batch_size

        # TODO: is there a better way to let the user control the device rather than it being the device of the n_atoms tensor?
        device = n_atoms.device

        # remove COM from ligand positions
        node_batch_idx = torch.arange(batch_size, device=g.device)
        node_batch_idx = node_batch_idx.repeat_interleave(g.batch_num_nodes())
        g = self.remove_com(g, node_batch_idx)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.n_timesteps)):
            s_arr = torch.full(size=(batch_size,), fill_value=s, device=device)
            t_arr = s_arr + 1
            s_arr = s_arr / self.n_timesteps
            t_arr = t_arr / self.n_timesteps

            g = self.sample_p_zs_given_zt(s_arr, t_arr, g, node_batch_idx)

        lig_pos = []
        lig_feat = []
        g = g.to('cpu')
        for g_i in dgl.unbatch(g):
            lig_pos.append(g_i.ndata['x_0'])
            lig_feat.append(g_i.ndata['h_0'])

        return lig_pos, lig_feat