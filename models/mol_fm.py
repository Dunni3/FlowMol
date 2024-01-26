from typing import Dict, List
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

from data_processing.utils import build_edge_idxs, get_upper_edge_mask, get_batch_idxs
from analysis.molecule_builder import SampledMolecule
from analysis.metrics import SampleAnalyzer

class MolFM(pl.LightningModule):

    canonical_feat_order = ['x', 'a', 'c', 'e']
    node_feats = ['x', 'a', 'c']
    edge_feats = ['e']

    def __init__(self,
                 atom_type_map: List[str],               
                 batches_per_epoch: int,
                 n_atoms_hist_file: str,
                 n_atom_charges: int = 6,
                 n_bond_types: int = 5,
                 sample_interval: float = 1.0, # how often to sample molecules from the model, measured in epochs
                 n_mols_to_sample: int = 64, # how many molecules to sample from the model during each sample/eval step during training
                 time_scaled_loss: bool = True,
                 position_prior_std: float = 1.0,
                 total_loss_weights: Dict[str, float] = {}, 
                 lr_scheduler_config: dict = {},
                 interpolant_scheduler_config: dict = {},
                 vector_field_config: dict = {}
                 ):
        super().__init__()

        self.batches_per_epoch = batches_per_epoch
        self.lr_scheduler_config = lr_scheduler_config
        self.atom_type_map = atom_type_map
        self.n_atom_types = len(atom_type_map)
        self.n_atom_charges = n_atom_charges
        self.n_bond_types = n_bond_types
        self.total_loss_weights = total_loss_weights
        self.time_scaled_loss = time_scaled_loss
        self.position_prior_std = position_prior_std

        # create a dictionary mapping feature -> number of categories
        self.n_cat_dict = {
            'a': self.n_atom_types,
            'c': n_atom_charges,
            'e': n_bond_types,
        }

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
        self.vector_field = GVPVectorField(n_atom_types=self.n_atom_types, n_charges=n_atom_charges, n_bond_types=n_bond_types,
                                           **vector_field_config)

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

        self.sample_interval = sample_interval # how often to sample molecules from the model, measured in epochs
        self.n_mols_to_sample = n_mols_to_sample # how many molecules to sample from the model during each sample/eval step during training
        self.last_sample_marker = 0 # this is the epoch_exact value of the last time we sampled molecules from the model
        self.sample_analyzer = SampleAnalyzer()


        # record the last epoch value for training steps -  this is really hacky but it lets me
        # align the validation losses with the correspoding training epoch value on W&B
        self.last_epoch_exact = 0

        self.save_hyperparameters()

    def training_step(self, g: dgl.DGLGraph, batch_idx: int):

        # compute epoch as a float
        epoch_exact = self.current_epoch + batch_idx/self.batches_per_epoch
        self.last_epoch_exact = epoch_exact

        # update the learning rate
        self.lr_scheduler.step_lr(epoch_exact)

        # sample and evaluate molecules if necessary
        if epoch_exact - self.last_sample_marker >= self.sample_interval:
            self.last_sample_marker = epoch_exact
            self.eval()
            with torch.no_grad():
                sampled_molecules = self.sample_random_sizes(n_molecules=self.n_mols_to_sample, device=g.device, n_timesteps=100)
            self.train()
            sampled_mols_metrics = self.sample_analyzer.analyze(sampled_molecules)
            self.log_dict(sampled_mols_metrics)

        # compute losses
        losses = self(g)

        # create a dictionary of values to log
        train_log_dict = {}
        train_log_dict['epoch_exact'] = epoch_exact

        for key in losses:
            train_log_dict[f'{key}_train_loss'] = losses[key]

        total_loss = torch.zeros(1, device=g.device, requires_grad=True)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]

        self.log_dict(train_log_dict, sync_dist=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True, sync_dist=True)

        return total_loss
    
    def validation_step(self, g: dgl.DGLGraph, batch_idx: int):
        # compute losses
        losses = self(g)

        # create dictionary of values to log
        val_log_dict = {
            'epoch_exact': self.last_epoch_exact
        }

        for key in losses:
            val_log_dict[f'{key}_val_loss'] = losses[key]

        self.log_dict(val_log_dict, batch_size=g.batch_size, sync_dist=True)

        # combine individual losses into a total loss
        total_loss = torch.zeros(1, device=g.device, requires_grad=False)
        for feat in self.canonical_feat_order:
            total_loss = total_loss + self.total_loss_weights[feat]*losses[feat]

        self.log('val_total_loss', total_loss, prog_bar=True, batch_size=g.batch_size, on_step=True, sync_dist=True)

        return total_loss
    
    def forward(self, g: dgl.DGLGraph):
        
        batch_size = g.batch_size
        device = g.device

        # get batch indicies of every atom and edge
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # create a mask which selects all of the upper triangle edges from the batched graph
        upper_edge_mask = get_upper_edge_mask(g)

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
        dst_dict = self.vector_field.pred_dst(g, t, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)

        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            # get the target for this feature
            if feat == 'e':
                target = g.edata[f'{feat}_1_true'][upper_edge_mask]
            else:
                target = g.ndata[f'{feat}_1_true']

            # get the target as class indicies for categorical features
            if feat in ['a', 'c', 'e']:
                target = target.argmax(dim=-1)

            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == 'e':
                    weight = weight[edge_batch_idx]
                else:
                    weight = weight[node_batch_idx]
                weight = weight.unsqueeze(-1)
            else:
                weight = 1.0

            # compute the losses
            losses[feat] = self.loss_fn_dict[feat](dst_dict[feat], target)*weight

            # when time_scaled_loss is True, we set the reduction to 'none' so that each training example can be scaled by the time-dependent weight.
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
        num_nodes = g.num_nodes()
        device = g.device
        g.ndata['x_0'] = torch.randn(num_nodes, 3, device=device)*self.position_prior_std
        g.ndata['x_0'] = g.ndata['x_0'] - dgl.readout_nodes(g, feat='x_0', op='mean')[node_batch_idx]

        # sample atom types, charges from simplex
        # TODO: can we implement OT flow matching for the simplex prior?
        for node_feat in ['a', 'c']:

            # get the number of cateogories for this feature
            n_cats = self.n_cat_dict[node_feat]

            # sample from simplex prior
            g.ndata[f'{node_feat}_0'] = self.exp_dist.sample((num_nodes, n_cats)).to(device)
            g.ndata[f'{node_feat}_0'] = g.ndata[f'{node_feat}_0'] / g.ndata[f'{node_feat}_0'].sum(dim=1, keepdim=True)

        # sample bond types from simplex prior - making sure the sample for the lower triangle is the same as the upper triangle
        n_edges = g.num_edges()
        n_upper_edges = n_edges // 2
        g.edata['e_0'] = torch.zeros(n_edges, self.n_bond_types, device=device).float()
        e_0_upper = self.exp_dist.sample((n_upper_edges, self.n_bond_types)).to(device)
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

            src_weight, dst_weight = src_weights[:, feat_idx][node_batch_idx].unsqueeze(-1), dst_weights[:, feat_idx][node_batch_idx].unsqueeze(-1)
            g.ndata[f'{feat}_t'] = src_weight * g.ndata[f'{feat}_0'] + dst_weight * g.ndata[f'{feat}_1_true']

        e_idx = self.canonical_feat_order.index('e')
        src_weight, dst_weight = src_weights[:, e_idx][edge_batch_idx].unsqueeze(-1), dst_weights[:, e_idx][edge_batch_idx].unsqueeze(-1)
        g.edata[f'e_t'] = src_weight * g.edata[f'e_0'] + dst_weight * g.edata[f'e_1_true']

        return g

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr_scheduler_config['base_lr'])
        self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.lr_scheduler_config)
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

    def sample_random_sizes(self, n_molecules: int, device="cuda:0", n_timesteps: int = 20, visualize=False):
        """Sample n_moceules with the number of atoms sampled from the distribution of the training set."""

        # get the number of atoms that will be in each molecules
        atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)

        return self.sample(atoms_per_molecule, n_timesteps=n_timesteps, device=device, visualize=visualize)
    
    def integrate(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor, upper_edge_mask: torch.Tensor, n_timesteps: int, visualize=False):
        """Integrate the trajectories of molecules along the vector field."""

        # get the timepoint for integration
        t = torch.linspace(0, 1, n_timesteps, device=g.device)

        # get the corresponding alpha values for each timepoint
        alpha_t = self.interpolant_scheduler.alpha_t(t)
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)

        # set x_t = x_0
        for feat in self.node_feats:
            g.ndata[f'{feat}_t'] = g.ndata[f'{feat}_0']

        for feat in self.edge_feats:
            g.edata[f'{feat}_t'] = g.edata[f'{feat}_0']


        # if visualizing the trajectory, create a datastructure to store the trajectory
        if visualize:
            traj_frames = {}
            for feat in self.canonical_feat_order:
                if feat == "e":
                    data_src = g.edata
                    split_sizes = g.batch_num_edges()
                else:
                    data_src = g.ndata
                    split_sizes = g.batch_num_nodes()

                split_sizes = split_sizes.detach().cpu().tolist()
                init_frame = data_src[f'{feat}_0'].detach().cpu()
                init_frame = torch.split(init_frame, split_sizes)
                traj_frames[feat] = [ init_frame ]

        for s_idx in range(1,t.shape[0]):

            # get the next timepoint (s) and the current timepoint (t)
            s_i = t[s_idx]
            t_i = t[s_idx - 1]
            alpha_t_i = alpha_t[s_idx - 1]
            alpha_t_prime_i = alpha_t_prime[s_idx - 1]

            # predict the destination of the trajectory given the current timepoint
            dst_dict = self.vector_field.pred_dst(
                g, 
                t=torch.full((g.batch_size,), t_i, device=g.device),
                node_batch_idx=node_batch_idx,
                upper_edge_mask=upper_edge_mask,
                apply_softmax=True
            )

            # compute x_s for each feature and set x_t = x_s
            for feat_idx, feat in enumerate(self.canonical_feat_order):
                x1_weight = alpha_t_prime_i[feat_idx]*(s_i - t_i)/(1 - alpha_t_i[feat_idx])
                xt_weight = 1 - x1_weight

                if feat == "e":
                    g_data_src = g.edata

                    # set the edge features so that corresponding upper and lower triangle edges have the same value
                    x1 = torch.zeros_like(g.edata['e_0'])
                    x1[upper_edge_mask] = dst_dict[feat]
                    x1[~upper_edge_mask] = dst_dict[feat]
                else:
                    g_data_src = g.ndata
                    x1 = dst_dict[feat]

                g_data_src[f'{feat}_t'] = x1_weight*x1 + xt_weight*g_data_src[f'{feat}_t']

                if visualize:
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    if feat == 'e':
                        split_sizes = g.batch_num_edges()
                    else:
                        split_sizes = g.batch_num_nodes()
                    split_sizes = split_sizes.detach().cpu().tolist()
                    frame = g_data_src[f'{feat}_t'].detach().cpu()
                    frame = torch.split(frame, split_sizes)
                    traj_frames[feat].append(frame)

        # set x_1 = x_t
        for feat in self.canonical_feat_order:

            if feat == "e":
                g_data_src = g.edata
            else:
                g_data_src = g.ndata

            g_data_src[f'{feat}_1'] = g_data_src[f'{feat}_t']

        if visualize:

            # currently, traj_frames[key] is a list of lists. each sublist contains the frame for every molecule in the batch
            # we want to rearrange this so that traj_frames is a list of dictionaries, where each dictionary contains the frames for a single molecule
            n_frames = len(traj_frames['x'])
            reshaped_traj_frames = []
            for mol_idx in range(g.batch_size):
                molecule_dict = {}
                for feat in self.canonical_feat_order:
                    feat_traj = []
                    for frame_idx in range(n_frames):
                        feat_traj.append(traj_frames[feat][frame_idx][mol_idx])
                    molecule_dict[feat] = torch.stack(feat_traj)
                reshaped_traj_frames.append(molecule_dict)


            return g, reshaped_traj_frames
        
        return g
    

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, n_timesteps: int = 20, device="cuda:0", visualize=False):
        """Sample molecules with the given number of atoms.
        
        Args:
            n_atoms (torch.Tensor): Tensor of shape (batch_size,) containing the number of atoms in each molecule.
        """
        batch_size = n_atoms.shape[0]

        # get the edge indicies for each unique number of atoms
        edge_idxs_dict = {}
        for n_atoms_i in torch.unique(n_atoms):
            edge_idxs_dict[int(n_atoms_i)] = build_edge_idxs(n_atoms_i)

        # construct a graph for each molecule
        g = []
        for n_atoms_i in n_atoms:
            edge_idxs = edge_idxs_dict[int(n_atoms_i)]
            g_i = dgl.graph((edge_idxs[0], edge_idxs[1]), num_nodes=n_atoms_i, device=device)
            g.append(g_i)
            

        # batch the graphs
        g = dgl.batch(g)

        # get upper edge mask
        upper_edge_mask = get_upper_edge_mask(g)

        # compute node_batch_idx
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # sample molecules from prior
        g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        # integrate trajectories
        itg_result = self.integrate(g, node_batch_idx, upper_edge_mask=upper_edge_mask, n_timesteps=n_timesteps, visualize=visualize)

        if visualize:
            g, traj_frames = itg_result
        else:
            g = itg_result

        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')


        molecules = []
        for mol_idx, g_i in enumerate(dgl.unbatch(g)):

            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args))

        return molecules
