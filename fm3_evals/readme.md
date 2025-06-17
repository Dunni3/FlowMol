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

And then these commands from `fm3_evals/ablations`:
```console
python gen_cmds/gen_min_cmds.py kek_runs/ --n_cpus=16 --cmd_file=cmd_files/min_cmds.txt
sbatch --array 1-4 slurm_files/min.slurm cmd_files/min_cmds.txt
```

## Baslines
Baselines (2) is more quirky. This is because sampling each model is different, so we decided the starting point for baselines should just be like an sdf file or a pickle file with rdkit molecules in it.

1. collect a directory of sdf files or rdkit pickles with molecules in them
2. run a script to generate commands that will call `fm3_evals/baselines/compute_baseline_comparison.py` on each file in the directory, and write metrics out to some file (which?)


## Geometry eval

Copying some stuff from Filipp's readme here:

---

### 3. `energy_benchmark/xtb_optimization.py`

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

### 5. `energy_benchmark/rmsd_energy.py`

**Purpose:** Compute GFN2-xTB energy gains, MMFF energy drops, and RMSD across molecule pairs.

**Usage:**
```bash
python energy_benchmark/rmsd_energy.py \
  --init_sdf path/to/initial.sdf \
  --opt_sdf path/to/optimized.sdf \
  --n_subsets 5
```

---