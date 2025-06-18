this is my hairball collection of scripts to run evaluations for the flowmol3 paper. largely, the approach is split into two sections:

1. ablations
2. baselines

# Ablations
Ablations (1) requires just sampling from trained flowmol models. For this we can just use the test.py script, except for the geometry analysis. 

Ablations is intended to contain like, training runs from flowmol where we changed features. We actually have two sets of ablations we need to analyze, and I think maybe only one will be in the paper.

The first ablation was basically just: should we kekulize our molecules or not?
The other ablations will remove some of our like, imporant features, to demonstrate their importance. 


The steps here are: 
1. collect a directory of flowmol model directories.
2. run `ablations/gen_test_cmds.py` to generate a set of commands to run. this will produce something like `test_cmds.sh` or a file you can set with `--cmd_file` at the command line. The file that `gen_test_cmds.py` produces will be a list of commands to run `test.py` on each model in the directory.
3. run the commands in `test_cmds.sh` to sample from the models. 
4. do geometry analysis on the sampled molecules.


I ran this command:
```console
python gen_test_cmds.py --models_root=kek_runs/ --n_timesteps=250 --metrics --n_mols=5000 --reos_raw --n_subsets=5
```

## how we did the geometry analysis on the sampled molecules  

And then these commands from `fm3_evals/ablations`:
```console
python gen_cmds/gen_min_cmds.py kek_runs/ --n_cpus=16 --cmd_file=cmd_files/min_cmds.txt
sbatch --array 1-4 slurm_files/min.slurm cmd_files/min_cmds.txt
python gen_cmds/gen_rmsd_cmds.py kek_runs/ --cmd_file=cmd_files/rmsd_cmds.txt --n_subsets=5
sbatch --array 1-4 slurm_files/rmsd.slurm cmd_files/rmsd_cmds.txt
```

## Baslines
Baselines (2) is more quirky. This is because sampling each model is different, so we decided the starting point for baselines should just be like an sdf file or a pickle file with rdkit molecules in it.

1. collect a directory of sdf files or rdkit pickles with molecules in them
2. run `fm3_evals/baselines/gen_cmds/gen_baseline_comparison_cmds.py` to generate commands that will call `fm3_evals/baselines/compute_baseline_comparison.py` on each file in the directory, and write metrics out to some file, perhaps like the input file name with `_metrics.json` appended to it.

I ran these commands from `fm3_evals/baselines`:
```console
python gen_cmds/gen_baseline_comparison_cmds.py baseline_mols/ --cmd_file=cmd_files/fm_metrics.txt --n_subsets=5 --reos_raw --dataset=geom_5_aromatic
sbatch --array 3-7 slurm_files/onecpu.slurm cmd_files/fm_metrics.txt
```

An annoying gotcha here is that for the flowmol model here i had to manually go in and set `--dataset=geom_5_kekulized` in the command file and add `--kekulized` ot the its invocation of `compute_baseline_comparison.py` because the flowmol model is kekulized, but the baseline molecules are not. 


## Geometry analysis on baseline models

This has yet to be done.

## todo

1. the baseline eval script is breaking because of boron
2. have not yet run geometry eval but i decided that the baseline eval should be writing results to a different directory than the one the molecules themselves are stored in, as this will also possibly be a more useful pattern for the geometry eval


# Geometry eval

Ok so the geometry eval proceeds through two scripts, run sequentially, described below in the order they are used:

---

## `fm3_evals/geometry/xtb_optimization.py`

**Purpose:** Optimize molecules with GFN2-xTB and extract energy/RMSD values.

**Usage:**
```bash
MKL_NUM_THREADS=16 OMP_NUM_THREADS=16 python energy_benchmark/xtb_optimization.py \
  --input_sdf path/to/generated.sdf \
  --output_sdf path/to/optimized_output.sdf \
  --init_sdf path/to/saved_initial_structures.sdf
```

**Note:** Requires `xtb` to be installed and available in your `PATH`.

---

## `fm3_evals/geometry/rmsd_energy.py`

**Purpose:** Compute GFN2-xTB energy gains, MMFF energy drops, and RMSD across molecule pairs.

**Usage:**
```bash
python energy_benchmark/rmsd_energy.py \
  --init_sdf path/to/initial.sdf \
  --opt_sdf path/to/optimized.sdf \
  --n_subsets 5
```

---