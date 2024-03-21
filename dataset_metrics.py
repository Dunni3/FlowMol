import argparse
from pathlib import Path
import yaml
from model_utils.load import data_module_from_config, read_config_file
from analysis.molecule_builder import SampledMolecule
from analysis.metrics import SampleAnalyzer
from utils.divergences import save_reference_dist
from typing import List
import numpy as np
import pickle
from tqdm import tqdm
from data_processing.utils import get_upper_edge_mask

# disable rdkit logging
from rdkit.Chem import AllChem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def parse_args():
    p = argparse.ArgumentParser(description='Dataset Metrics')
    p.add_argument('--config', type=Path, required=True, help='Path to config file')
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

    # convert the training dataset to a list of SampledMolecule objects
    mols = dataset_to_mols(train_dataset, config['dataset']['atom_map'])

    # compute the UFF energies for the dataset
    energies = sample_analyzer.compute_sample_energy(mols)

    # compute a discrete distribution of energies
    bins = np.linspace(-200, 500, 200)
    counts_dataset, _ = np.histogram(energies, bins=bins, density=False)
    p_dataset = counts_dataset / counts_dataset.sum()

    # save the reference distribution
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    energy_dist_file = processed_data_dir / 'energy_dist.npz'
    save_reference_dist(bins, p_dataset, energy_dist_file)

    # compute metrics on the sampled molecules
    metrics = sample_analyzer.analyze(mols)
    metrics_file = processed_data_dir / 'metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    # print metrics
    for k, v in metrics.items():
        print(f'{k}= {v:.2f}')