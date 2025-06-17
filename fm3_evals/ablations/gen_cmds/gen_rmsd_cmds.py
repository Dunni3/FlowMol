import argparse
from pathlib import Path
from flowmol.utils.path import flowmol_root

def parse_args():
    p = argparse.ArgumentParser(description="Generate geometry commands for XTB optimization for a directory of model directories.")
    p.add_argument('models_dir', type=Path, help="Directory model directories.")
    p.add_argument('--init_mols_name', type=str, default='xtb_init_mols.sdf', help='Name of the initial molecules SDF file.')
    p.add_argument('--min_mols_name', type=str, default='xtb_minimized_mols.sdf', help='Name of the minimized molecules SDF file.')
    p.add_argument('--cmd_file', type=Path, default=Path('rmsd_cmds.txt'))
    p.add_argument('--n_subsets', type=int, default=1, help='Number of subsets to compute std over.')
    # p.add_argument('--n_cpus', type=int, default=1, help='Number of cpus to use for xtb optimization.')

    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    script_path = Path(flowmol_root()) / 'fm3_evals/geometry/rmsd_energy.py'

    for model_dir in args.models_dir.resolve().iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        samples_dir = model_dir / 'samples'
        init_mols_file = samples_dir / args.init_mols_name
        min_mols_file = samples_dir / args.min_mols_name

        if not init_mols_file.exists() or not min_mols_file.exists():
            print(f"Required files {init_mols_file} or {min_mols_file} do not exist, skipping model {model_name}.")
            continue

        cmd = f'python {script_path} --init_sdf {init_mols_file} --opt_sdf {min_mols_file} --n_subsets={args.n_subsets}\n'

        with open(args.cmd_file, 'a') as f:
            f.write(cmd)