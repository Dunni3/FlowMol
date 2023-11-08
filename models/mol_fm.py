from typing import Dict
import torch
import torch.optim as optim
import pytorch_lightning as pl
import dgl
import torch.nn.functional as fn

from .lr_scheduler import LRScheduler
from .interpolant_scheduler import InterpolantScheduler
from .vector_field import GVPVectorField

class MolFM(pl.LightningModule):

    def __init__(self,
                 n_atom_types: int,                  
                 batches_per_epoch: int,
                 n_atoms_hist_file: str,
                 n_atom_charges: int = 6,
                 n_bond_types: int = 5, 
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
        
        # construct histogram of number of atoms in each ligand
        self.n_atoms_hist_file = n_atoms_hist_file
        self.build_n_atoms_dist(n_atoms_hist_file=n_atoms_hist_file)

        # create interpolant scheduler and vector field
        self.interpolant_scheduler = InterpolantScheduler(**interpolant_scheduler_config)
        self.vector_field = GVPVectorField(**vector_field_config)

        self.save_hyperparameters()

    def training_step(self, g: dgl.DGLGraph, batch_idx: int):

        epoch_exact = self.current_epoch + batch_idx/self.batches_per_epoch
        self.sched.step_lr(epoch_exact)
        losses = self(g)

        self.log('epoch_exact', epoch_exact, on_step=True, prog_bar=True)

        for key in losses:
            if key == 'l2':
                continue
            self.log(key, losses[key], on_step=True, prog_bar=True)

        return losses['l2']
    
    def forward(self, g: dgl.DGLGraph):
        
        batch_size = g.batch_size
        device = g.device

        # get batch indicies of every ligand
        node_batch_idx = torch.arange(batch_size, device=device)
        node_batch_idx = node_batch_idx.repeat_interleave(g.batch_num_nodes())

        # sample timepoints for each item in the batch
        t = torch.randint(0, self.n_timesteps, size=(batch_size,), device=device).float() # timesteps
        t = t / self.n_timesteps

        # sample epsilon for each ligand
        eps = {
            'h':torch.randn(g.ndata['h_0'].shape, device=device),
            'x':torch.randn(g.ndata['x_0'].shape, device=device)
        }

        # construct noisy versions of the ligand
        gamma_t = self.gamma(t).to(device=device)
        g = self.noised_representation(g, node_batch_idx, eps, gamma_t)

        # predict the noise that was added
        eps_h_pred, eps_x_pred = self.dynamics(g, t, node_batch_idx)

        x_loss = (eps['x'] - eps_x_pred).square().sum()
        h_loss = (eps['h'] - eps_h_pred).square().sum()

        losses = {
            'l2_x': x_loss/eps_x_pred.numel(),
            'l2_h': h_loss/eps_h_pred.numel(),
            'l2': (x_loss + h_loss)/( eps_x_pred.numel() + eps_h_pred.numel() )
        }
        return losses
    
    def remove_com(self, g: dgl.DGLGraph, batch_idx: torch.Tensor):
        com = dgl.readout_nodes(g, feat='x_0', op='mean')
        raise NotImplementedError
        g.ndata['x_0'] = g.ndata['x_0'] - com[batch_idx]
        return g

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.scheduler_config['base_lr'])
        self.sched = Scheduler(model=self, optimizer=optimizer, **self.scheduler_config)
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