import argparse
from pathlib import Path
from flowmol.utils.path import flowmol_root

def parse_args():
    p = argparse.ArgumentParser(description="Generate geometry commands for XTB optimization for a directory of model directories.")
    p.add_argument('baselines_dir', type=Path, help="Directory model directories.")
    p.add_argument('--inits_dir', type=Path, default=Path('xtb_inits'),
                   help="Directory to store initial structures for xtb optimization.")
    p.add_argument('--mins_dir', type=Path, default=Path('xtb_mins'),
                   help="Directory to store minimized structures from xtb optimization.")
    p.add_argument('--cmd_file', type=Path, default=Path('geometry_cmds.txt'))
    p.add_argument('--n_cpus', type=int, default=1, help='Number of cpus to use for xtb optimization.')

    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    minimization_script = Path(flowmol_root()) / 'fm3_evals/geometry/xtb_optimization.py'

    if not args.inits_dir.exists():
        args.inits_dir.mkdir(parents=True, exist_ok=True)
    if not args.mins_dir.exists():
        args.mins_dir.mkdir(parents=True, exist_ok=True)

    cmds = []
    for sample_file in args.baselines_dir.resolve().iterdir():
        if not sample_file.is_file():
            continue


        init_sdf = args.inits_dir / f'xtb_init_{sample_file.stem}.sdf'
        output_sdf = args.mins_dir / f'xtb_min_{sample_file.stem}.sdf'

        cmds.append(
            f'MKL_NUM_THREADS={args.n_cpus} OMP_NUM_THREADS={args.n_cpus} python {minimization_script} --input_sdf {sample_file} --output_sdf {output_sdf} --init_sdf {init_sdf}\n'
        )

    with open(args.cmd_file, 'w') as f:
        f.writelines(cmds)
    print(f"Wrote {len(cmds)} commands to {args.cmd_file}.")
