import argparse
from pathlib import Path
from flowmol.utils.path import flowmol_root

def parse_args():
    p = argparse.ArgumentParser(description="Generate geometry commands for XTB optimization for a directory of model directories.")
    p.add_argument('baselines_dir', type=Path, help="Directory of molecules obtained from baseline models.")
    p.add_argument('--init_mols_dir', type=Path, default=None, help='Directory of the initial molecules SDF files.')
    p.add_argument('--min_mols_dir', type=Path, default=None, help='Directory of the minimized molecules SDF files.')
    p.add_argument('--cmd_file', type=Path, default=Path('rmsd_cmds.txt'))
    p.add_argument('--n_subsets', type=int, default=1, help='Number of subsets to compute std over.')
    # p.add_argument('--n_cpus', type=int, default=1, help='Number of cpus to use for xtb optimization.')

    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    script_path = Path(flowmol_root()) / 'fm3_evals/geometry/rmsd_energy.py'

    cmds = []
    for sample_file in args.baselines_dir.resolve().iterdir():
        if not sample_file.is_file():
            continue

        # find the file with initial structure 
        if args.init_mols_dir is None:
            init_mols_file = args.baselines_dir.parent / 'xtb_inits' / f'xtb_init_{sample_file.stem}.sdf'
        else:
            init_mols_file = args.init_mols_dir / f'xtb_init_{sample_file.stem}.sdf'
        
        # find the file with minimized structure
        if args.min_mols_dir is None:
            min_mols_file = args.baselines_dir.parent / 'xtb_mins' / f'xtb_min_{sample_file.stem}.sdf'
        else:
            min_mols_file = args.min_mols_dir / f'xtb_min_{sample_file.stem}.sdf'

        if not init_mols_file.exists() or not min_mols_file.exists():
            print(f"Required files {init_mols_file} or {min_mols_file} do not exist, skipping.")
            continue

        output_file = args.baselines_dir.parent / 'baseline_results' / f'{sample_file.stem}_rmsd_energy_results.pkl'

        cmd = f'python {script_path} --init_sdf {init_mols_file} --opt_sdf {min_mols_file} --n_subsets={args.n_subsets} --output_file={output_file}'

        cmds.append(cmd)
    with open(args.cmd_file, 'w') as f:
        f.write('\n'.join(cmds)+'\n')