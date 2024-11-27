from setuptools import setup, find_packages

setup(
    name="flowmol",
    version="0.1",
    description="Flow matching for 3D de novo molecule generation",
    author="Ian Dunn",
    author_email="ian.dunn@pitt.edu",
    python_requires=">=3.10,<3.11",
    packages=find_packages(),
    install_requires=[
        "torch @ https://download.pytorch.org/whl/cu121/torch-2.2.0%2Bcu121-cp310-cp310-linux_x86_64.whl",
        "torchvision @ https://download.pytorch.org/whl/cu121/torchvision",
        "torchaudio @ https://download.pytorch.org/whl/cu121/torchaudio",
        "torch-cluster==1.6.3",
        "torch-scatter==2.1.2",
        "dgl @ https://data.dgl.ai/wheels/torch-2.2/cu121/dgl-2.2.0%2Bcu121-cp310-cp310-manylinux1_x86_64.whl",
        "pytorch-lightning==2.1.3",
        "rdkit==2023.09.4",
        "wandb",
        "einops",
        "pystow",
        "useful_rdkit_utils",
    ],
)