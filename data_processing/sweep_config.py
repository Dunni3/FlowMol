import argparse
from distutils.util import strtobool

def register_hyperparameter_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register hyperparameter arguments for the model."""

    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--warmup_length', type=float, default=None)

    # hyperparams which can be set for each feature
    feats = ['x', 'a', 'c', 'e']
    for feat in feats:
        p.add_argument(f'--{feat}_loss_weight', type=float, default=None)
        p.add_argument(f'--{feat}_cos_param', type=float, default=None, help='cosine parameter for interpolation schedule')

    # prior parameters
    p.add_argument('--ot_node_feats', type=str, default=None, help='whether to solve assignment problem on node features, true or false')
    p.add_argument('--rotate_positions', type=str, default=None, help='whether to rotate positions for prior alignment, true or false')
    p.add_argument('--positions_std', type=float, default=None, help='standard deviation of prior distribution for positions')
    p.add_argument('--biased_edge_prior', type=str, default=None, help='whether to use a biased edge prior, true or false')
    p.add_argument('--no_bond_prob', type=float, default=None, help='probability of no bond for biased edge prior')
    p.add_argument('--bond_order_std', type=float, default=None, help='standard deviation of noising for biased edge prior')

    # vector field configs
    p.add_argument('--n_vec_channels', type=int, default=None, help='number of vector features stored on each node')
    p.add_argument('--n_hidden_scalars', type=int, default=None, help='number of scalar features stored on each node')
    p.add_argument('--n_hidden_edge_feats', type=int, default=None, help='number of scalar features stored on each edge')
    p.add_argument('--n_recycles', type=int, default=None, help='number of times to recycle vector field convolutions')
    p.add_argument('--n_molecule_updates', type=int, default=None, help='number of times to update molecule features per recycle')
    p.add_argument('--separate_mol_updaters', type=str, default=None, help='whether to use separate position/edge update modules for each update, true or false')
    p.add_argument('--convs_per_update', type=int, default=None, help='number of convolutions to perform per molecule update')
    p.add_argument('--n_cp_feats', type=int, default=None, help='number of cross-product features computed inside GVPs')
    p.add_argument('--n_message_gvps', type=int, default=None, help='number of message passing GVPs')
    p.add_argument('--n_update_gvps', type=int, default=None, help='number of update GVPs')
    p.add_argument('--message_norm', type=str, default=None, help='how to normalize messages, number or "mean"')
    p.add_argument('--rbf_dmax', type=float, default=None, help='maximum distance for RBF kernel')
    p.add_argument('--rbf_dim', type=int, default=None, help='dimension of RBF kernel')




def merge_config_and_args(config: dict, args: argparse.Namespace) -> dict:
    """Merge the model configuration with the command line arguments."""

    # hyperparameters for training
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    if args.lr is not None:
        config['lr_scheduler']['base_lr'] = args.lr

    if args.warmup_length is not None:
        config['lr_scheduler']['warmup_length'] = args.warmup_length

    # hyperparams which can be set for each feature
    feats = ['x', 'a', 'c', 'e']
    for feat in feats:
        if getattr(args, f'{feat}_loss_weight') is not None:
            config['mol_fm']['total_loss_weights'][feat] = getattr(args, f'{feat}_loss_weight')

        if getattr(args, f'{feat}_cos_param') is not None:
            config['interpolant_scheduler']['cosine_params'][feat] = getattr(args, f'{feat}_cos_param')

    # prior parameters which are boolean
    for arg in ['ot_node_feats', 'rotate_positions', 'biased_edge_prior']:
        if getattr(args, arg) is not None:
            config['mol_fm']['prior_config'][arg] = strtobool(getattr(args, arg))

    # prior parameters which are numeric
    for arg in ['positions_std', 'no_bond_prob', 'bond_order_std']:
        if getattr(args, arg) is not None:
            config['mol_fm']['prior_config'][arg] = getattr(args, arg)

    # vector field configs which are boolean
    for arg in ['separate_mol_updaters']:
        if getattr(args, arg) is not None:
            config['vector_field'][arg] = strtobool(getattr(args, arg))

    # vector field configs which are numeric
    for arg in ['n_vec_channels', 'n_hidden_scalars', 'n_hidden_edge_feats', 'n_recycles', 'n_molecule_updates', 'convs_per_update', 'n_cp_feats', 'n_message_gvps', 'n_update_gvps', 'rbf_dim']:
        if getattr(args, arg) is not None:
            config['vector_field'][arg] = getattr(args, arg)

    # message norm parameter
    if args.message_norm is not None:
        message_norm = args.message_norm
        if message_norm.isnumeric():
            message_norm = float(message_norm)
        config['vector_field']['message_norm'] = message_norm

    return config
    

