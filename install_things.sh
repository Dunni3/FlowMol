#!/bin/bash

# activate the conda environemnt
# source /home/ian/mambaforge/etc/profile.d/mamba.sh
# source /home/ian/mambaforge/etc/profile.d/conda.sh

# get the name of the current conda environment
ENV_NAME=$(basename "$CONDA_PREFIX")

# print the name of the current conda environment to the terminal
echo "Installing things into the environment '$ENV_NAME'"

set -e
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install pytorch-cluster pytorch-scatter -c pyg -y
mamba install -c dglteam/label/cu118 dgl -y
mamba install -c conda-forge pytorch-lightning -y
mamba install -c conda-forge openff-toolkit -y
mamba install -c conda-forge rdkit -y
