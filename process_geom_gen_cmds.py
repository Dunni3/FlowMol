import argparse
from pathlib import Path
import numpy as np
import json
import yaml
import pickle

def parse_args():
    """Parse command line arguments using argparse."""
    p = argparse.ArgumentParser(description='Process geometry')

    p.add_argument('split_file', type=Path, help='path to split file')
    p.add_argument('--config', type=Path, help='config file path')

    p.add_argument('--start_idx', type=int, default=0, help='start index')
    p.add_argument('--end_idx', type=int, default=np.inf, help='end index')

    p.add_argument('--n_cpus', type=int, default=1, help='number of cpus to use when computing partial charges for confomers')
    p.add_argument('--chunk_size', type=int, default=2000, help='number of molecules to process at a time')

    p.add_argument('--mols_per_job', type=int, help='number of molecules per chunk', default=None)
    p.add_argument('--n_jobs', type=int, help='number of chunks that the dataset should be split into', default=None)

    p.add_argument('--output_file', type=Path, default=Path('process_geom_cmds.sh'), help='path to output file')

    # p.add_argument('--overwrite', action='store_true', help='overwrite existing files')
    # p.add_argument('--save_interval', type=int, default=5, help='number of molecules after which to save processed data')

    args = p.parse_args()

    # check that start_idx is before end_idx
    if args.start_idx >= args.end_idx:
        raise ValueError(f"start_idx must be less than end_idx")
    
    # check that either n_jobs or mols_per_job is not None and throw an error if they are both None
    if args.n_jobs is None and args.mols_per_job is None:
        raise ValueError(f"either n_jobs or mols_per_job must be specified")
    
    if args.n_jobs is not None and args.mols_per_job is not None:
        raise ValueError(f"only one of n_jobs or mols_per_job can be specified")

    return args


if __name__ == '__main__':
    args = parse_args()

    # process config file into dictionary
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the raw data file
    raw_data_file = args.split_file

    # load the raw data
    with open(raw_data_file, 'rb') as f:
        raw_data = pickle.load(f)

    # get size of the full dataset
    n_molecules = sum( len(mol_list[1]) for mol_list in raw_data )
    del raw_data

    # determine if the dataset size is truncated as specified by the config file
    if config['dataset']['dataset_size'] is not None:
        assert config['dataset']['dataset_size'] <= n_molecules, f"dataset_size cannot be larger than the full dataset size"
        n_molecules = config['dataset']['dataset_size']

    if args.n_jobs is not None:
        # check that n_chunks is less than or equal to the number of molecules in the dataset
        assert args.n_jobs <= n_molecules, f"n_jobs cannot be larger than the full dataset size"

        # get number of molecules per chunk
        args.mols_per_job = int(np.ceil(n_molecules / args.n_chunks))
    else:
        # check that mols_per_chunk is less than or equal to the number of molecules in the dataset
        assert args.mols_per_job <= n_molecules, f"mols_per_job cannot be larger than the full dataset size"

    # get start and end indices for each chunk, where chunks are non-overlapping subsets of the full dataset
    start_indices = np.arange(0, n_molecules, args.mols_per_job)
    end_indices = np.arange(args.mols_per_job, n_molecules + args.mols_per_job, args.mols_per_job)
    
    # if the last chunk as currently computed is too large, reduce the size of the last chunk
    if end_indices[-1] > n_molecules:
        end_indices[-1] = n_molecules

    # construct a command line command for each chunk
    cmds = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        cmd = f"python process_geom.py {args.split_split} --config {args.config} --start_idx {start_idx} --end_idx {end_idx} --n_cpus {args.n_cpus}\n"
        cmds.append(cmd)

    # write commands to file
    with open(args.output_file, 'w') as f:
        f.writelines(cmds)