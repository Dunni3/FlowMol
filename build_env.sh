#!/bin/bash

# activate the conda environemnt
mamba activate flowmol

# get the name of the current conda environment
ENV_NAME=$(basename "$CONDA_PREFIX")

# print the name of the current conda environment to the terminal
echo "Installing things into the environment '$ENV_NAME'"

set -e
mamba install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba install pytorch-cluster=1.6.3 pytorch-scatter=2.1.2 -c pyg -y
mamba install -c dglteam/label/cu121 dgl=2.0 -y
mamba install -c conda-forge pytorch-lightning=2.1.0 -y
mamba install -c conda-forge rdkit=2023.09.1 pystow -y
pip install wandb useful_rdkit_utils --no-input
