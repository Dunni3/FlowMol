import argparse
from pathlib import Path
from flowmol.utils.path import flowmol_root

def parse_args():
    p = argparse.ArgumentParser(description="Generate geometry commands for XTB optimization for a directory of model directories.")
    p.add_argument('models_dir', type=Path, help="Directory model directories.")
    p.add_argument('--sample_file_name', type=str, default='sampled_mols.sdf')
    p.add_argument('--output_file_name', type=str, default=None)
    p.add_argument('--cmd_file', type=Path, default=Path('geometry_cmds.txt'))
    p.add_argument('--n_cpus', type=int, default=1, help='Number of cpus to use for xtb optimization.')

    args = p.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    minimization_script = Path(flowmol_root()) / 'fm3_evals/geometry/xtb_optimization.py'

    cmds = []
    for model_dir in args.models_dir.resolve().iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        samples_dir = model_dir / 'samples'
        sample_file = samples_dir / args.sample_file_name
        if not sample_file.exists():
            print(f"Sample file {sample_file} does not exist, skipping model {model_name}.")
            continue
        
        if args.output_file_name is None:
            output_file_name = "xtb_minimized_mols.sdf"
            
        else:
            output_file_name = args.output_file_name
        init_mols_file_name = "xtb_init_mols.sdf"
        output_file = samples_dir / output_file_name
        init_mols_file = samples_dir / init_mols_file_name

        cmds.append(
            f'MKL_NUM_THREADS={args.n_cpus} OMP_NUM_THREADS={args.n_cpus} python {minimization_script} --input_sdf {sample_file} --output_sdf {output_file} --init_sdf {init_mols_file}\n'
        )

    with open(args.cmd_file, 'w') as f:
        f.writelines(cmds)
    print(f"Wrote {len(cmds)} commands to {args.cmd_file}.")
