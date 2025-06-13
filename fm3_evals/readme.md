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
python gen_test_cmds.py --models_root=kek_runs/ --n_timesteps=250 --metrics --n_mols=5000 --reos_raw
```

## Baslines
Baselines (2) is more quirky. This is because sampling each model is different, so we decided the starting point for baselines should just be like an sdf file or a pickle file with rdkit molecules in it.


## Geometry eval