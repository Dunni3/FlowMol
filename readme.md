# Mixed Continuous and Categorical Flow Matching for 3D De Novo Molecule Generation

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat)](https://arxiv.org/abs/2404.19739)

![Image](figures/ga.png)

This is the offical implementation of the paper [Mixed Continuous and Categorical Flow Matching for 3D De Novo Molecule Generation](https://arxiv.org/abs/2404.19739).

# Environment Setup

Create a conda/mamba envrionment with python 3.10: `mamba create -n flowmol python=3.10`

After activating the envrionment we just created, setup the environment by running the bash script `build_env.sh`. This script just runs a handful of install commands. The script uses mamba by default. If you have conda installed just change the `mamba` command to `conda` in the script.

# How we define a model (config files)

Specifications of the model and the data that the model is trained on are all packaged into one config file. The config files are just yaml files. Once you setup your config file, you pass it as input to the data processing scripts in addition to the training scripts. An example config file is provided at `configs/dev.yml`. This example config file also has some helpful comments in it describing what the different parameters mean.

Actual config files used to train models presented in the paper are available in the `trained_models/` directory.

Note, you don't have to reprocess the dataset for every model you train, as long as the models you are training contain the same parameters under the `dataset` section of the config file. 

# Downloading Trained Models

Run the following command to download trained models:

```console
wget -r -np -nH --cut-dirs=2 --reject 'index.html*' https://bits.csb.pitt.edu/files/FlowMol/trained_models/
```

Trained models will now be available within the `trained_models/` directory. Checkout the [trained models readme](trained_models/readme.md) for more information on the trained models.

# Sampling

To sample from a trained model, use the `test.py` script and pass a model checkpoint with the `--checkpoint` argument. Here's an example command to sample from a trained model:

```console
python test.py --checkpoint=trained_models/qm9_gaussian/checkpoints/model.ckpt --n_mols=1000 --n_timesteps=100 --max_batch_size=500 --output_file=brand_new_molecules.sdf
```

The output file, if specified, must be an SDF file. If not specified, sampled molecules will be written to the model directory. You can also have the script produce a molecule for every integration step to see the evolution of the molecule over time by adding the `--visualize` flag. You can compute all of the metrics reported in the paper by adding the `--metrics` flag.

# Datasets

Our workflow for datasets is:
1. download the raw dataset
2. process the dataset using one of the `process_<dataset>.py` scripts. these scripts accept a config file as input. You can use one of the config files packaged with the trained models in the `trained_models/` directory.
3. now you will be able to train a model using the processed dataset, as long as the dataset configuration in the config file you use to train the model matches the dataset configuration in the config file you used to process the dataset.

## QM9

Starting from the root of this repository, run these commands to download the raw qm9 dataset:
```console
mkdir data/qm9_raw
cd data/qm9_raw
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
wget -O uncharacterized.txt https://ndownloader.figshare.com/files/3195404
unzip qm9.zip
```

You can run this command to process the qm9 dataset:
```console
python process_qm9.py --config=trained_models/qm9_gauss_ep/config.yml
```

## GEOM-Drugs

Starting from the root of this repository, run `mkdir data/geom_raw`. Download the train, validation, and test splits [provided by MiDi](https://github.com/cvignac/MiDi#datasets) into the `data/geom_raw` directory. Then, from the root of this repository, run these commands to process the geom dataset:
```console
python process_geom.py data/geom_raw/train_data.pickle --config=trained_models/geom_gauss_ep/config.yml
python process_geom.py data/geom_raw/test_data.pickle --config=trained_models/geom_gauss_ep/config.yml
python process_geom.py data/geom_raw/val_data.pickle --config=trained_models/geom_gauss_ep/config.yml
```

# Training

Run the `train.py` script. You can either pass a config file, or you can pass a trained model checkpoint for resuming. Note in the latter case, the script assumes the checkpoint is inside of a directory that contains a config file. To see the expected file structure of a model directory, refer to the [trained models readme](trained_models/readme.md). Here's an example command to train a model:

```console
python train.py --config=trained_models/qm9_gaussian/config.yaml
```
