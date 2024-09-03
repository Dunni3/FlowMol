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

    return p.parse_args()

def dataset_to_mols(train_dataset, atom_type_map) -> List[SampledMolecule]:
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
        sampled_mol = SampledMolecule(g, atom_type_map)
        mols.append(sampled_mol)

    return mols


if __name__ == "__main__":

    args = parse_args()

    # read config file
    config: dict = read_config_file(args.config)

    # get the training dataset
    data_module = data_module_from_config(config)

    # run setup so that the data module creates dataset classes
    data_module.setup(stage='fit')

    # get training dataset
    train_dataset = data_module.train_dataset

    # create sample analyzer
    sample_analyzer = SampleAnalyzer()

    if args.n_mols is not None:
        # randomly select n_mols numbers from the range (0, len(dataset))
        indices = np.random.choice(len(train_dataset), args.n_mols, replace=False)
        train_dataset = [train_dataset[i] for i in indices]
    
    # compute metrics
    if args.batch_size:
        n_batches = math.ceil(len(train_dataset) / args.batch_size)
        energies = []
        batch_metrics = []
        for i in range(n_batches):
            start = i * args.batch_size
            end = min(start + args.batch_size, len(train_dataset))
            mols = [ train_dataset[dataset_idx] for dataset_idx in range(start, end)]
            mols = dataset_to_mols(mols, config['dataset']['atom_map'])
            energies.extend(sample_analyzer.compute_sample_energy(mols))
            batch_metrics.append(sample_analyzer.analyze(mols, return_counts=True))

        # compute metrics on the sampled molecules
        numerators = defaultdict(float)
        denominators = defaultdict(float)
        for count_dict in batch_metrics:
            numerators['frac_atoms_stable'] += count_dict['n_stable_atoms']
            numerators['frac_mols_stable_valence'] += count_dict['n_stable_molecules']
            numerators['frac_valid_mols'] += count_dict['n_valid']
            numerators['avg_frag_frac'] += count_dict['sum_frag_fracs']
            numerators['avg_num_components'] += count_dict['sum_num_components']

            denominators['frac_atoms_stable'] += count_dict['n_atoms']
            denominators['frac_mols_stable_valence'] += count_dict['n_molecules']
            denominators['frac_valid_mols'] += count_dict['n_molecules']
            denominators['avg_frag_frac'] += count_dict['n_frag_fracs']
            denominators['avg_num_components'] += count_dict['n_num_components']
        
        metrics = {}
        for key in numerators.keys():
            metrics[key] = numerators[key] / denominators[key]

    else:
        # convert the training dataset to a list of SampledMolecule objects
        mols = dataset_to_mols(train_dataset, config['dataset']['atom_map'])

        # compute the energies for the dataset
        energies = sample_analyzer.compute_sample_energy(mols)

        # compute metrics on the sampled molecules
        metrics = sample_analyzer.analyze(mols)

    # compute a discrete distribution of energies
    bins = np.linspace(-200, 500, 200) # this range of bins captures ~99% of the density for the MMFF energies of both QM9 and GEOM-DRUGS datasets -- is that reasonable?
    counts_dataset, _ = np.histogram(energies, bins=bins, density=False)
    # compute the fraction of the molecules which fall outside these bins
    frac_outside = 1 - counts_dataset.sum() / len(energies)
    # print the fraction of the molecules which fall outside the bins
    print(f'fraction of molecules outside the bins: {frac_outside:.4f}')
    p_dataset = counts_dataset / len(energies)

    # save the reference distribution
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    energy_dist_file = processed_data_dir / 'energy_dist.npz'
    save_reference_dist(bins, p_dataset, energy_dist_file)

    # write metrics
    metrics_file = processed_data_dir / 'metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    # print metrics
    for k, v in metrics.items():
        print(f'{k}= {v:.2f}')