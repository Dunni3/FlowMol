import argparse
import shlex
from pathlib import Path
from flowmol.utils.path import flowmol_root

def main():
    parser = argparse.ArgumentParser(
        description="Generate a shell script of compute_baseline_comparison.py commands for each sample file."
    )
    parser.add_argument(
        "samples_dir",
        type=Path,
        help="Directory containing sample files (SDF or PKL) to analyze."
    )
    parser.add_argument(
        "--cmd_file",
        type=Path,
        default=Path("baseline_comparison_cmds.sh"),
        help="Output path for the generated shell script."
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default=None,
        help="File pattern to match sample files (e.g., '*.pkl', '*.sdf'). If None, all files in the directory are included."
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
    )
    # all other args are passed through to compute_baseline_comparison.py
    args, passthrough = parser.parse_known_args()

    compute_script = Path(flowmol_root()) / 'fm3_evals/baselines/compute_baseline_comparison.py'
    out = args.cmd_file
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.output_dir is None:
        output_dir = args.samples_dir
    else:
        output_dir = args.output_dir
    with open(out, "w") as f:
        f.write("#!/usr/bin/env bash\n\n")
        # gather all sample files
        if args.file_pattern is None:
            sample_files = sorted([file for file in args.samples_dir.iterdir() if file.is_file()])
        else:
            sample_files = sorted(args.samples_dir.glob(args.file_pattern))
            
        for sample_file in sample_files:
            sample_file = sample_file.resolve()
            output_file = output_dir / (sample_file.stem + "_metrics.pkl")
            cmd_parts = ["python", 
                         shlex.quote(str(compute_script)),
                         shlex.quote(str(sample_file)),
                         f"--output_file={output_file}"
                        ]
            # append any extra flags/values passed to this script
            for tok in passthrough:
                cmd_parts.append(shlex.quote(tok))
            f.write(" ".join(cmd_parts) + "\n")

    # make the script executable
    try:
        out.chmod(0o755)
    except Exception:
        pass

    print(f"Wrote {len(sample_files)} commands to {out}")

if __name__ == "__main__":
    main()
