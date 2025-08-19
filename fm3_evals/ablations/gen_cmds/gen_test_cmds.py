import argparse
import shlex
from pathlib import Path
from flowmol.utils.path import flowmol_root

def main():
    parser = argparse.ArgumentParser(
        description="Generate a shell script of test.py commands for each model directory."
    )
    parser.add_argument(
        "--models_root",
        type=Path,
        required=True,
        help="Root directory containing one subdirectory per trained model."
    )
    parser.add_argument(
        "--cmd_file",
        type=Path,
        default=Path("test_cmds.sh"),
        help="Output path for the generated shell script."
    )
    # all other args are passed through to test.py
    args, passthrough = parser.parse_known_args()
    test_py = Path(flowmol_root()) / 'test.py'
    out = args.cmd_file
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write("#!/usr/bin/env bash\n\n")
        # gather all model subdirs
        model_dirs = sorted(d for d in args.models_root.iterdir() if d.is_dir())
        for md in model_dirs:
            md = md.resolve()
            cmd_parts = ["python", shlex.quote(str(test_py)),
                         "--model_dir", shlex.quote(str(md))]
            # append any extra flags/values passed to this script
            for tok in passthrough:
                cmd_parts.append(shlex.quote(tok))
            f.write(" ".join(cmd_parts) + "\n")

    # make the script executable
    try:
        out.chmod(0o755)
    except Exception:
        pass

    print(f"Wrote {len(model_dirs)} commands to {out}")

if __name__ == "__main__":
    main()