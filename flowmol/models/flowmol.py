from typing import Dict, List
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import dgl
import torch.nn.functional as fn
from torch.distributions import Exponential
from pathlib import Path

from flowmol.models.lr_scheduler import LRScheduler
from flowmol.models.interpolant_scheduler import InterpolantScheduler
from flowmol.models.vector_field import EndpointVectorField, VectorField, DirichletVectorField
from flowmol.models.ctmc_vector_field import CTMCVectorField

from flowmol.data_processing.utils import build_edge_idxs, get_upper_edge_mask, get_batch_idxs
from flowmol.data_processing.priors import uniform_simplex_prior, biased_simplex_prior, batched_rigid_alignment, rigid_alignment
from flowmol.data_processing.priors import inference_prior_register, edge_prior
from flowmol.analysis.molecule_builder import SampledMolecule
from flowmol.analysis.metrics import SampleAnalyzer
from einops import rearrange

class FlowMol(pl.LightningModule):

    canonical_feat_order = ['x', 'a', 'c', 'e']
    node_feats = ['x', 'a', 'c']
    edge_feats = ['e']

    def __init__(self,
                 atom_type_map: List[str],               
                 n_atoms_hist_file: str,
                 marginal_dists_file: str,                 
                 n_atom_charges: int = 6,
                 sample_interval: float = 1.0, # how often to sample molecules from the model, measured in epochs
                 n_mols_to_sample: int = 64, # how many molecules to sample from the model during each sample/eval step during training
                 time_scaled_loss: bool = True,
                 exclude_charges: bool = False,
                 weight_ae: bool = False, # whether or not to apply weights to the atom and edge losses (infrequent categories given more weight)
                 target_blur: float = 0.0, # how much to blur the target distribution for categorical features
                 parameterization: str = 'endpoint', # how to parameterize the flow-matching objective, can be 'endpoint', 'vector-field', or 'dirichlet'
                 total_loss_weights: Dict[str, float] = {}, 
                 lr_scheduler_config: dict = {},
                 interpolant_scheduler_config: dict = {},
                 vector_field_config: dict = {},
                 prior_config: dict = {},
                 default_n_timesteps: int = 250,
                 ema_weight: float = 0.999, # TODO: currently unused but thought i implemented it at some point? maybe floating in a branch somewhere
                 fake_atom_p: float = 0.0,
                 distort_p: float = 0.0,
                 explicit_aromaticity: bool = False,
                 ):
        super().__init__()

        self.lr_scheduler_config = lr_scheduler_config
        self.atom_type_map = atom_type_map
        self.n_atom_types = len(atom_type_map)
        self.n_atom_charges = n_atom_charges
        self.n_bond_types = 5 if explicit_aromaticity else 4
        self.total_loss_weights = total_loss_weights
        self.time_scaled_loss = time_scaled_loss
        self.prior_config = prior_config
        self.exclude_charges = exclude_charges
        self.marginal_dists_file = marginal_dists_file
        self.parameterization = parameterization
        self.weight_ae = weight_ae
        self.target_blur = target_blur
        self.n_atoms_hist_file = n_atoms_hist_file
        self.default_n_timesteps = default_n_timesteps
        self.distort_p = distort_p
        self.explicit_aromaticity = explicit_aromaticity

        # fake atoms settings
        self.fake_atom_p = fake_atom_p
        self.fake_atoms = fake_atom_p > 0
        if self.fake_atoms:
            self.n_atom_types += 1

        if self.weight_ae and parameterization == 'vector-field':
            raise NotImplementedError('weighting the atom and edge losses is not yet implemented for the vector-field parameterization')
        
        if self.target_blur != 0.0 and parameterization == 'vector-field':
            raise NotImplementedError('blurring the target distribution is not yet implemented for the vector-field parameterization')
        
        if self.target_blur < 0.0:
            raise ValueError('target_blur must be non-negative')
        
        # if provided filepath to data dir does not exist, assume it is relative to the repo root
        processed_data_dir = Path(self.marginal_dists_file).parent
        if not processed_data_dir.exists():
            repo_root = Path(__file__).parent.parent.parent
            processed_data_dir = repo_root / processed_data_dir
            self.marginal_dists_file = processed_data_dir / self.marginal_dists_file.name
            self.n_atoms_hist_file = processed_data_dir / self.n_atoms_hist_file.name

        # do some boring stuff regarding the prior distribution
        self.configure_prior()

        if self.exclude_charges:
            self.node_feats.remove('c')
            self.canonical_feat_order.remove('c')
            self.total_loss_weights.pop('c')

        # create a dictionary mapping feature -> number of categories
        n_bond_types = 5 if self.explicit_aromaticity else 4
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
        self.build_n_atoms_dist(n_atoms_hist_file=self.n_atoms_hist_file)

        # create interpolant scheduler and vector field
        self.interpolant_scheduler = InterpolantScheduler(canonical_feat_order=self.canonical_feat_order, 
                                                          **interpolant_scheduler_config)
        
        # check that a valid parameterization was specified
        if self.parameterization not in ['endpoint', 'vector-field', 'dirichlet', 'ctmc']:
            raise ValueError(f'parameterization must be one of "endpoint", "vector-field", or "dirichlet", "ctmc", got {self.parameterization}')

        if self.parameterization == 'endpoint':
            vector_field_class = EndpointVectorField
        elif self.parameterization == 'vector-field':
            vector_field_class = VectorField
        elif self.parameterization == 'dirichlet':
            vector_field_class = DirichletVectorField
        elif self.parameterization == 'ctmc':
            vector_field_class = CTMCVectorField

        self.vector_field = vector_field_class(n_atom_types=self.n_atom_types,
                                           canonical_feat_order=self.canonical_feat_order,
                                           interpolant_scheduler=self.interpolant_scheduler, 
                                           n_charges=n_atom_charges, 
                                           n_bond_types=n_bond_types,
                                           exclude_charges=self.exclude_charges,
                                           fake_atoms=self.fake_atoms,
                                           **vector_field_config)

        # remove charge loss function if necessary
        if self.exclude_charges:
            self.loss_fn_dict.pop('c')

        self.sample_interval = sample_interval # how often to sample molecules from the model, measured in epochs
        self.n_mols_to_sample = n_mols_to_sample # how many molecules to sample from the model during each sample/eval step during training
        self.last_sample_marker = 0 # this is the epoch_exact value of the last time we sampled molecules from the model
        self.sample_analyzer = SampleAnalyzer(processed_data_dir=processed_data_dir)


        # record the last epoch value for training steps -  this is really hacky but it lets me
        # align the validation losses with the correspoding training epoch value on W&B
        self.last_epoch_exact = 0

        self.save_hyperparameters()

    def configure_prior(self):
        # load the marginal distributions of atom types, bond orders and the conditional distribution of charges given atom type
        p_a, p_c, p_e, p_c_given_a = torch.load(self.marginal_dists_file)
        self.p_a = p_a
        self.p_e = p_e

        # add the marginal distributions as arguments to the prior sampling functions
        if self.prior_config['a']['type'] == 'marginal':
            self.prior_config['a']['kwargs']['p'] = p_a

        if self.prior_config['e']['type'] == 'marginal':
            self.prior_config['e']['kwargs']['p'] = p_e

        if self.prior_config['c']['type'] == 'marginal':
            self.prior_config['c']['kwargs']['p'] = p_c
        
        if self.prior_config['c']['type'] == 'c-given-a':
            self.prior_config['c']['kwargs']['p_c_given_a'] = p_c_given_a

        if self.parameterization == 'dirichlet':
            for feat in ['a', 'c', 'e']:
                if self.prior_config[feat]['type'] != 'uniform-simplex':
                    raise ValueError('dirichlet parameterization requires that all categorical priors be uniform-simplex')

        if self.parameterization == 'ctmc':
            for feat in ['a', 'c', 'e']:
                if self.prior_config[feat]['type'] != 'ctmc':
                    raise ValueError('ctmc parameterization requires that all categorical priors be ctmc')
                
    def configure_loss_fns(self, device):    
        # instantiate loss functions
        if self.time_scaled_loss:
            reduction = 'none'
        else:
            reduction = 'mean'

        if self.parameterization in  ['endpoint', 'dirichlet', 'ctmc']:
            categorical_loss_fn = nn.CrossEntropyLoss
        elif self.parameterization == 'vector-field':
            categorical_loss_fn = nn.MSELoss


        if self.weight_ae:
            a_kwargs = {'weight': (1 - self.p_a).to(device)}
            e_kwargs = {'weight': (1 - self.p_e).to(device)}
        else:
            a_kwargs = {}
            e_kwargs = {}

        if self.parameterization == 'ctmc':
            cat_kwargs = {'ignore_index': -100}
        else:
            cat_kwargs = {}

        self.loss_fn_dict = {
            'x': nn.MSELoss(reduction=reduction),
            'a': categorical_loss_fn(reduction=reduction, **a_kwargs, **cat_kwargs),
            'c': categorical_loss_fn(reduction=reduction, **cat_kwargs),
            'e': categorical_loss_fn(reduction=reduction, **e_kwargs, **cat_kwargs),
        }

    def training_step(self, g: dgl.DGLGraph, batch_idx: int):

        # check if self has the attribute batches_per_epoch
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = len(self.trainer.train_dataloader)

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
                sampled_molecules = self.sample_random_sizes(n_molecules=self.n_mols_to_sample, device=g.device)
            self.train()
            sampled_mols_metrics = self.sample_analyzer.analyze(sampled_molecules, energy_div=False, functional_validity=True)
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

        # check if the attribute loss_fn_dict exists
        # it is necessary to do this here (as opposed to in __init__) beacause
        # to instantiate the loss function class-conditioned weights, the weight
        # tensors need to be on the same device as the graph...seems pretty dumb but that's how it is
        if not hasattr(self, 'loss_fn_dict'):
            self.configure_loss_fns(device=g.device)

        # get batch indicies of every atom and edge
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)

        # create a mask which selects all of the upper triangle edges from the batched graph
        upper_edge_mask = get_upper_edge_mask(g)

        # get initial COM of each molecule and remove the COM from the atom positions
        # this step is now done in MoleculeDataset.__getitem__ method, a necessary adjustment to do OT alignment on
        # the prior during the __getitem__ method
        # init_coms = dgl.readout_nodes(g, feat='x_1_true', op='mean')
        # g.ndata['x_1_true'] = g.ndata['x_1_true'] - init_coms[node_batch_idx]

        # sample molecules from prior
        # we used to sample the prior in the forward pass at training time,
        # but now at training time we sample the prior in the __getitem__ method of MoleculeDataset
        # this is so that we can compute OT alignments in parallel (since they cannot be done in batch)
        # g = self.sample_prior(g, node_batch_idx, upper_edge_mask)

        # sample timepoints for each molecule in the batch
        t = torch.rand(batch_size, device=device).float()

        # construct interpolated molecules
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx, edge_batch_idx, upper_edge_mask)

        if self.distort_p > 0.0:
            t_mask = (t > 0.5)[node_batch_idx]
            distort_mask = torch.rand(g.num_nodes(), 1, device=device) < self.distort_p
            distort_mask = distort_mask & t_mask.unsqueeze(-1)
            g.ndata['x_t'] = g.ndata['x_t'] + torch.randn_like(g.ndata['x_t'])*distort_mask*0.5

        # forward pass for the vector field
        vf_output = self.vector_field(g, t, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)

        # get the target (label) for each feature
        targets = {}
        alpha_t_prime = self.interpolant_scheduler.alpha_t_prime(t)
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata

            # compute the target for endpoint parameterization
            if self.parameterization in ['endpoint', 'dirichlet', 'ctmc']:
                target = data_src[f'{feat}_1_true']
                if feat == "e":
                    target = target[upper_edge_mask]
                if feat in ['a', 'c', 'e']:
                    if self.target_blur == 0.0:
                        target = target.argmax(dim=-1)
                    else:
                        target = target + torch.randn_like(target)*self.target_blur
                        target = fn.softmax(target, dim=-1)
            #  compute the target for vector-field parameterization
            elif self.parameterization == 'vector-field':
                alpha_t_prime_i = alpha_t_prime[:, feat_idx]
                x_1 = data_src[f'{feat}_1_true']
                x_0 = data_src[f'{feat}_0']

                if feat == 'e':
                    alpha_t_prime_i = alpha_t_prime_i[edge_batch_idx][upper_edge_mask].unsqueeze(-1)
                    x_1 = x_1[upper_edge_mask]
                    x_0 = x_0[upper_edge_mask]
                else:
                    alpha_t_prime_i = alpha_t_prime_i[node_batch_idx].unsqueeze(-1)

                target = alpha_t_prime_i*(x_1 - x_0)

            # for CTMC parameterization, we do not apply loss on already unmasked features
            if self.parameterization == 'ctmc' and feat in ['a', 'c', 'e']:
                if feat == 'e':
                    xt_idxs = data_src[f'{feat}_t'][upper_edge_mask].argmax(-1)
                else:
                    xt_idxs = data_src[f'{feat}_t'].argmax(-1)
                # note that we use the default ignore_index of the CrossEntropyLoss class here
                target[ xt_idxs != self.n_cat_dict[feat] ] = -100 # set the target to ignore_index when the feature is already unmasked in xt

            targets[feat] = target

        # get the time-dependent loss weights if necessary
        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)
            
        # compute losses
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):

            if self.time_scaled_loss:
                weight = time_weights[:, feat_idx]
                if feat == 'e':
                    weight = weight[edge_batch_idx][upper_edge_mask]
                else:
                    weight = weight[node_batch_idx]
                weight = weight.unsqueeze(-1)
            else:
                weight = 1.0

            # compute the losses
            target = targets[feat]
            losses[feat] = self.loss_fn_dict[feat](vf_output[feat], target)*weight

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
        num_nodes = g.num_nodes()
        device = g.device

        
        # sample the prior for node features
        for feat in self.node_feats:
            prior_type = self.prior_config[feat]['type']
            prior_fn = inference_prior_register[prior_type]
            # I tried to design consistent interface for prior functions, but it's not perfect
            # hence the need for the following two if statements
            if feat == 'x':
                args = [g, node_batch_idx,]
            else:
                args = [num_nodes, self.n_cat_dict[feat],]

            if feat == 'c' and self.prior_config[feat]['type'] == 'c-given-a':
                args.append(g.ndata['a_0'])

            kwargs = self.prior_config[feat]['kwargs']
            g.ndata[f'{feat}_0'] = prior_fn(*args, **kwargs).to(device)

        # sample the prior for edge features
        g.edata['e_0'] = edge_prior(upper_edge_mask, self.prior_config['e']).to(device)
            
        return g
    

    def configure_optimizers(self):
        try:
            weight_decay = self.lr_scheduler_config['weight_decay']
        except KeyError:
            weight_decay = 0

        optimizer = optim.Adam(self.parameters(), lr=self.lr_scheduler_config['base_lr'], weight_decay=weight_decay)
        self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **self.lr_scheduler_config)
        return optimizer

    def build_n_atoms_dist(self, n_atoms_hist_file: str):
        """Builds the distribution of the number of atoms in a ligand."""
        n_atoms, n_atom_counts = torch.load(n_atoms_hist_file)
        n_atoms_prob = n_atom_counts / n_atom_counts.sum()
        self.n_atoms_dist = torch.distributions.Categorical(probs=n_atoms_prob)
        self.n_atoms_map = n_atoms

    def sample_n_atoms(self, n_molecules: int, **kwargs):
        """Draw samples from the distribution of the number of atoms in a ligand."""
        n_atoms = self.n_atoms_dist.sample((n_molecules,), **kwargs)
        return self.n_atoms_map[n_atoms]

    def sample_random_sizes(self, n_molecules: int, device="cuda:0",
    stochasticity=None, high_confidence_threshold=None, 
    xt_traj=False, ep_traj=False, **kwargs):
        """Sample n_moceules with the number of atoms sampled from the distribution of the training set."""

        # get the number of atoms that will be in each molecules
        atoms_per_molecule = self.sample_n_atoms(n_molecules).to(device)

        return self.sample(atoms_per_molecule, 
            device=device,  
            stochasticity=stochasticity, 
            high_confidence_threshold=high_confidence_threshold,
            xt_traj=xt_traj,
            ep_traj=ep_traj, **kwargs)
    

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, n_timesteps: int = None, device="cuda:0",
        stochasticity=None, high_confidence_threshold=None, xt_traj=False, ep_traj=False, **kwargs):
        """Sample molecules with the given number of atoms.
        
        Args:
            n_atoms (torch.Tensor): Tensor of shape (batch_size,) containing the number of atoms in each molecule.
        """
        if n_timesteps is None:
            n_timesteps = self.default_n_timesteps

        if xt_traj or ep_traj:
            visualize = True
        else:
            visualize = False

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
        integrate_kwargs = {
            'upper_edge_mask': upper_edge_mask,
            'n_timesteps': n_timesteps,
            'visualize': visualize
        }
        if self.parameterization == 'ctmc':
            integrate_kwargs['stochasticity'] = stochasticity
            integrate_kwargs['high_confidence_threshold'] = high_confidence_threshold

        itg_result = self.vector_field.integrate(g, node_batch_idx, **integrate_kwargs, **kwargs)

        if visualize:
            g, traj_frames = itg_result
        else:
            g = itg_result

        g.edata['ue_mask'] = upper_edge_mask
        g = g.to('cpu')

        if self.parameterization == 'ctmc':
            ctmc_mol = True
        else:
            ctmc_mol = False


        molecules = []
        for mol_idx, g_i in enumerate(dgl.unbatch(g)):

            args = [g_i, self.atom_type_map]
            if visualize:
                args.append(traj_frames[mol_idx])

            molecules.append(SampledMolecule(*args, 
                ctmc_mol=ctmc_mol, 
                fake_atoms=self.fake_atoms,
                build_xt_traj=xt_traj,
                build_ep_traj=ep_traj,
                exclude_charges=self.exclude_charges,
                explicit_aromaticity=self.explicit_aromaticity,
            ))

        return molecules
