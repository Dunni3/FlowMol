# TODO:
- [ ] create trained models directory
- [ ] change geom config files so that raw data dir is data/geom_raw
- [ ] record versions of all things installed in the project
- [ ] cite pytorch and DGL in the paper
- [ ] upload trained models somewhere and write script for downloading them

# getting the data


# processing a split of the geom dataset

```console
python process_geom.py data/raw/test_data.pickle --config configs/dev.yml
```

# Environment Setup

Create a conda/mamba envrionment with python 3.10: `mamba create -n flowmol python=3.10`

After activating the envrionment we jsut created, setup the environment by running the bash script `install_things.sh`

This repository was built using the following versions of software:
- python 3.10.0
- pytorch 2.?.0
- dgl 0.7.2 ... these are wrongish versions they were just filled in by copilot

# Trained models

Download trained models by running this bash script from the root of the repository:  

# Project Structure

## How we define a model (config files)

Specifications of the model and the data that the model is trained on are all packaged into one config file. The config files are just yaml files. Once you setup your config file, you pass it as input to the data processing scripts in addition to the training scripts. An example config file is provided at `configs/dev_config.yml`. This example config file also has some helpful comments in it describing what the different parameters mean.

Actual config files used to train models presented in the paper are available in the `trained_models/` directory.

Note, you don't have to reprocess the dataset for every model you train, as long as the models you are training contain the same parameters under the `dataset` section of the config file. 

## A note on understanding our scripts

All of the steps of training, sampling, and evaluation are run through various scripts in this repo. In this readme, I describe in words the inputs that are provided to each script. Each of these scripts implements command line arguments via the argparse library. You can always run `python <script_name>.py --help` to see a list of command line arguments that the script accepts. You can also just open the script and inspect the `parse_args()` function to see what command line arguments are accepted.

# Datasets

Our workflow for datasets is:
1. download the raw dataset
2. process the dataset using one of the `processa_<dataset>.py` scripts. these scripts accept a config file as input. You can use one of the config files packaged with the trained models in the `trained_models/` directory.
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

