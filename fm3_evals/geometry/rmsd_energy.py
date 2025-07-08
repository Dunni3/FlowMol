import argparse
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import pickle

from geom_utils.utils import is_valid, compute_mmff_energy_drop, compute_rmsd

RDLogger.DisableLog('rdApp.*')


def compute_metrics_for_pairs(pairs, hydrogens=True):
    energy_gains, mmff_drops, rmsds = [], [], []

    output_counter = defaultdict(int)

    for init_mol, opt_mol in pairs:
        if init_mol is None or opt_mol is None or not is_valid(init_mol):
            continue

        try:
            energy_gain = float(opt_mol.GetProp('energy_gain')) if opt_mol.HasProp('energy_gain') else None
            rmsd = compute_rmsd(init_mol, opt_mol, hydrogens=hydrogens)
            mmff_drop = compute_mmff_energy_drop(init_mol)

            if energy_gain is not None:
                energy_gains.append(-energy_gain)
            else:
                output_counter['missing_energy_gain'] += 1

            if rmsd is not None:
                rmsds.append(rmsd)
            else:
                output_counter['missing_rmsd'] += 1

            if mmff_drop is not None:
                mmff_drops.append(mmff_drop)
            else:
                output_counter['missing_mmff_drop'] += 1

            # if energy_gain is not None and rmsd is not None:
            #     energy_gains.append(-energy_gain)
            #     rmsds.append(rmsd)
            #     if mmff_drop is not None:
            #         mmff_drops.append(mmff_drop)

            output_counter['successful_pairs'] += 1
        except Exception as e:
            print(e)
            continue

    # this was for debugging
    # print(*[f"{k}: {v}" for k, v in output_counter.items()], sep="\n", flush=True)

    return {
        "avg_energy_gain": np.mean(energy_gains) if energy_gains else 0.0,
        "med_energy_gain": np.median(energy_gains) if energy_gains else 0.0,
        "avg_rmsd": np.mean(rmsds) if rmsds else 0.0,
        "med_rmsd": np.median(rmsds) if rmsds else 0.0,
        "avg_mmff_drop": np.mean(mmff_drops) if mmff_drops else 0.0,
        "med_mmff_drop": np.median(mmff_drops) if mmff_drops else 0.0,
        "n": len(energy_gains)
    }


def split_into_subsets(pairs, n_subsets):
    """
    Split pairs into n_subsets with approximately equal size.
    """
    subset_size = len(pairs) // n_subsets
    return [pairs[i * subset_size:(i + 1) * subset_size] for i in range(n_subsets)]


def main():
    parser = argparse.ArgumentParser(description="Compute energy and RMSD metrics from SDF files.")
    parser.add_argument("--init_sdf", required=True, help="Path to initial SDF file")
    parser.add_argument("--opt_sdf", required=True, help="Path to optimized SDF file")
    parser.add_argument("--no_hydrogens", action="store_true", help="Remove hydrogens before computing RMSD")
    parser.add_argument("--n_subsets", type=int, default=1, help="Number of subsets to compute std over")
    parser.add_argument('--output_file', type=str, default=None, help="Output file to save results (optional)")

    args = parser.parse_args()
    hydrogens = not args.no_hydrogens

    init_mols = [mol for mol in Chem.SDMolSupplier(args.init_sdf, sanitize=False, removeHs=False) if mol]
    opt_mols = [mol for mol in Chem.SDMolSupplier(args.opt_sdf, sanitize=False, removeHs=False) if mol]

    assert len(init_mols) == len(opt_mols), "Initial and optimized files must contain same number of molecules."

    pairs = list(zip(init_mols, opt_mols))

    if args.n_subsets <= 1:
        results = compute_metrics_for_pairs(pairs, hydrogens=hydrogens)
        print(f"Processed {results['n']} molecules.")
        for key, val in results.items():
            if key != "n":
                print(f"{key.replace('_', ' ').title()}: {val:.4f}")

        output_results = results
    else:
        subset_metrics = defaultdict(list)
        subsets = split_into_subsets(pairs, args.n_subsets)

        for subset in subsets:
            result = compute_metrics_for_pairs(subset, hydrogens=hydrogens)
            for key, value in result.items():
                subset_metrics[key].append(value)

        n_total = sum(subset_metrics["n"])
        print(f"Processed {n_total} molecules across {args.n_subsets} subsets.")

        output_results = {}
        for key in ["avg_energy_gain", "med_energy_gain", "avg_rmsd", "med_rmsd", "avg_mmff_drop", "med_mmff_drop"]:
            values = subset_metrics[key]
            mean = np.mean(values)
            std = np.std(values)
            ci95 = 1.96 * (std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            print(f"{key.replace('_', ' ').title()}: {mean:.4f} ± {std:.4f}")
            output_results[key] = mean
            output_results[f"{key}_ci95"] = ci95

    if args.output_file is None:
        output_file_pkl = Path(args.init_sdf).parent / "rmsd_energy_results.pkl"
        output_file_txt = Path(args.init_sdf).parent / "rmsd_energy_results.txt"
    else:
        output_file_pkl = Path(args.output_file).with_suffix('.pkl')
        output_file_txt = Path(args.output_file).with_suffix('.txt')

    with open(output_file_pkl, 'wb') as f:
        pickle.dump(output_results, f)
    print(f"Results saved to {output_file_pkl}")
    with open(output_file_txt, 'w') as f:
        for key, value in output_results.items():
            f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
    print(f"Results saved to {output_file_txt}")
    



if __name__ == "__main__":
    main()
