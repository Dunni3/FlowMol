#!/bin/bash

# activate the conda environemnt
mamba activate flowmol

# get the name of the current conda environment
ENV_NAME=$(basename "$CONDA_PREFIX")

# print the name of the current conda environment to the terminal
echo "Installing things into the environment '$ENV_NAME'"

set -e
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba install pytorch-cluster pytorch-scatter -c pyg -y
mamba install -c dglteam/label/cu121 dgl -y
mamba install -c conda-forge pytorch-lightning -y
mamba install -c conda-forge rdkit -y
pip install wandb --no-input
