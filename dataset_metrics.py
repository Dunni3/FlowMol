import argparse
from pathlib import Path
import yaml
from flowmol.model_utils.load import data_module_from_config, read_config_file
from flowmol.analysis.molecule_builder import SampledMolecule
from flowmol.analysis.metrics import SampleAnalyzer
from flowmol.utils.divergences import save_reference_dist
from typing import List
import numpy as np
import pickle
from tqdm import tqdm
from flowmol.data_processing.utils import get_upper_edge_mask
import math
from collections import defaultdict

# disable rdkit logging
from rdkit.Chem import AllChem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def parse_args():
    p = argparse.ArgumentParser(description='Dataset Metrics')
    p.add_argument('--config', type=Path, required=True, help='Path to config file')
    p.add_argument('--n_mols', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--pb_workers', type=int, default=0, help='Number of workers for PoseBusters analysis')
    p.add_argument('--output_file', type=Path, default=None, help='Path to output metrics file (default: processed_data_dir/metrics_fm3.pkl)')

    return p.parse_args()

def dataset_to_mols(train_dataset, atom_type_map, fake_atoms) -> List[SampledMolecule]:
    print('converting dataset from graphs to molecules')
    mols = []
    for g in tqdm(train_dataset, mininterval=15):

        for feat in 'xace':
            if feat == 'e':
                data_src = g.edata
            else:
                data_src = g.ndata
            data_src[f'{feat}_1'] = data_src[f'{feat}_1_true']

        g.edata['ue_mask'] = get_upper_edge_mask(g)
        sampled_mol = SampledMolecule(g, atom_type_map, fake_atoms=fake_atoms)
        mols.append(sampled_mol)

    return mols


if __name__ == "__main__":

    args = parse_args()

    # read config file
    config: dict = read_config_file(args.config)

    # determine whether there are fake atoms
    fake_atoms = config['mol_fm'].get('fake_atom_p', 0.0) > 0.0

    # get the training dataset
    data_module = data_module_from_config(config)

    # run setup so that the data module creates dataset classes
    data_module.setup(stage='fit')

    # get training dataset
    train_dataset = data_module.train_dataset

    # create sample analyzer
    processed_data_dir = config['dataset']['processed_data_dir']
    processed_data_dir = Path(processed_data_dir)
    sample_analyzer = SampleAnalyzer(processed_data_dir=processed_data_dir, pb_energy=True, pb_workers=args.pb_workers)

    if args.n_mols is not None:
        # randomly select n_mols numbers from the range (0, len(dataset))
        indices = np.random.choice(len(train_dataset), args.n_mols, replace=False)
        train_dataset = [train_dataset[i] for i in indices]
    
    # compute metrics
    if args.batch_size:
        n_batches = math.ceil(len(train_dataset) / args.batch_size)
        batch_metrics = []
        for i in range(n_batches):
            start = i * args.batch_size
            end = min(start + args.batch_size, len(train_dataset))
            mols = [ train_dataset[dataset_idx] for dataset_idx in range(start, end)]
            mols = dataset_to_mols(mols, config['dataset']['atom_map'], fake_atoms=fake_atoms)
            # energies.extend(sample_analyzer.compute_sample_energy(mols))
            metrics = sample_analyzer.analyze(mols, posebusters=True)
            metrics['n_mols'] = len(mols)
            batch_metrics.append(metrics)

        # compute aggregated metrics
        metrics_to_aggregate = [metric_name for metric_name in metrics.keys() if 'pb_' in metric_name]
        metrics_to_aggregate.extend(['frac_valid_mols', 'frac_connected'])

        numerators = defaultdict(float)
        denominators = defaultdict(int)

        for metrics_dict in batch_metrics:
            for metric_name in metrics_to_aggregate:
                numerators[metric_name] += metrics_dict[metric_name]*metrics_dict['n_mols']
                denominators[metric_name] += metrics_dict['n_mols']
        
        output_metrics = { metric_name: numerators[metric_name]/denominators[metric_name] for metric_name in numerators }


    else:
        # convert the training dataset to a list of SampledMolecule objects
        mols = dataset_to_mols(train_dataset, config['dataset']['atom_map'])

        # compute metrics on the sampled molecules
        output_metrics = sample_analyzer.analyze(mols, posebusters=True)

    # save the reference distribution
    processed_data_dir = Path(config['dataset']['processed_data_dir'])

    # write metrics
    if args.output_file is not None:
        metrics_file = args.output_file
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        metrics_file = processed_data_dir / 'metrics_fm3.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(output_metrics, f)

    # print metrics
    for k, v in metrics.items():
        print(f'{k}= {v:.2f}')