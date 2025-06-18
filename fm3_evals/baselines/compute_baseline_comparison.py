from pathlib import Path
import argparse
import pickle
from flowmol.analysis.metrics import SampleAnalyzer
from flowmol.analysis.molecule_builder import SampledMolecule
from rdkit import Chem
import numpy as np
import math

def parse_args():
    p = argparse.ArgumentParser(description='Computes Metrics on a set of molecules sampled using a generative model.')
    p.add_argument('sample_file', type=Path, help='Path to file containing sampled molecules') # note this scripts assumes that the contents of the pickle file is a tuple where the first element is a list of rdkit molecules and the second element is the sampling time as a float in seconds
    p.add_argument('--output_file', type=Path, help='Path to output file', default=None)
    p.add_argument('--dataset', type=str, default=None, help='Name of dataset, can be geom or qm9')
    p.add_argument('--processed_data_dir', type=Path, default=None, help='Path to directory containing processed data for the dataset')
    p.add_argument('--n_subsets', type=int, default=None)
    p.add_argument('--reos_raw', action='store_true', help='whether to write out raw reos and rings data to file')
    p.add_argument('--kekulized', action='store_true', help='Whether the input molecules are kekulized.')

    args = p.parse_args()

    # if args.dataset and args.dataset not in ['geom', 'qm9']:
    #     raise ValueError('dataset must be one of [geom, qm9]')

    return args

if __name__ == "__main__":
    args = parse_args()

    # we used to assume that sample_file was a pickle file containing rdkit molecules 
    if args.sample_file.suffix == '.pkl':
        with open(args.sample_file, 'rb') as f:
            rdkit_mols, sampling_time = pickle.load(f)
    elif args.sample_file.suffix == '.sdf':
        supplier = Chem.SDMolSupplier(str(args.sample_file), sanitize=False, removeHs=False)
        rdkit_mols = [mol for mol in supplier]
        sampling_time = None
    else:
        raise ValueError

    sample_analyzer = SampleAnalyzer(dataset=args.dataset,)
    

    n_raw_samples = len(rdkit_mols)

    # convert rdkit molecules back to SampledMolecules
    sampled_mols = [ SampledMolecule.from_rdkit_mol(mol, explicit_aromaticity= not args.kekulized) for mol in rdkit_mols if mol is not None]

    if args.n_subsets is None:

        # compute metrics on the molecules
        metrics = sample_analyzer.analyze(sampled_mols,
            functional_validity=True,
            posebusters=True
        )
    else:
        mols_per_subset = len(rdkit_mols) // args.n_subsets
        subset_metrics = []
        for i in range(args.n_subsets):
            start_idx = i*mols_per_subset
            end_idx = min(start_idx + mols_per_subset, len(sampled_mols))
            subset_metrics.append(
                sample_analyzer.analyze(
                sampled_mols[start_idx:end_idx],
                functional_validity=True,
                posebusters=True
            ))

        metrics = {}
        for key in subset_metrics[0].keys():
            vals = np.array([d[key] for d in subset_metrics])
            mean = vals.mean()
            std = vals.std()
            count = vals.shape[0]
            ci95 = 1.96*std/math.sqrt(count)
            metrics[key] = mean
            metrics[f'{key}_ci95'] = ci95

    # add sampling time to metrics
    metrics['sampling_time'] = sampling_time


    # some of the initial samples in rdkit_mols could have been None if the original model was unable to convert model outputs -> rdkit molecule
    # we will correct for this in our computation of the fraction of valid molecules
    metrics['frac_valid_mols']= metrics['frac_valid_mols'] * len(sampled_mols) / n_raw_samples

    # write metrics to file
    if args.output_file is None:
        output_file = args.sample_file.parent / f'{args.sample_file.stem}_metrics.pkl'
    else:
        output_file = args.output_file
    
    with open(output_file, 'wb') as f:
        pickle.dump(metrics, f)


    # compute and write raw reos data
    if args.reos_raw:
        reos_raw = sample_analyzer.reos_and_rings(sampled_mols, return_raw=True)
        if args.output_file is None:
            reos_file = args.sample_file.parent / f'{args.sample_file.stem}_reos_and_rings.pkl'
        else:
            reos_file = args.output_file.parent / f'{args.output_file.stem}_reos_and_rings.pkl'
        with open(reos_file, mode='wb') as f:
            pickle.dump(reos_raw, f)
