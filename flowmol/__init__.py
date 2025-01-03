from pathlib import Path
from flowmol.models.flowmol import FlowMol
import subprocess

pretrained_model_names = [
    'geom_ctmc',
    'geom_gaussian',
    'geom_simplexflow',
    'geom_dirichlet',
    'qm9_ctmc',
    'qm9_gaussian',
    'qm9_simplexflow',
    'qm9_dirichlet'
]

def load_pretrained(model_name: str) -> FlowMol:
    """Load one of the pre-trained models by name.

    Args:
        model_name (str): Name of the model to load. Supported models are:
            'geom_ctmc',
            'geom_gaussian',
            'geom_simplexflow',
            'geom_dirichlet',
            'qm9_ctmc',
            'qm9_gaussian',
            'qm9_simplexflow',
            'qm9_dirichlet'  
    """
    if model_name not in pretrained_model_names:
        raise ValueError(f"Model {model_name} not found. Supported models: {pretrained_model_names}")

    model_dir = Path(__file__).parent / 'trained_models' / model_name

    if not model_dir.exists():
        # download the model if it doesn't exist
        download_remote_model_dir(model_dir)

    if 'qm9' in model_name:
        marginal_dists_file = Path(__file__).parent.parent / 'data' / 'qm9' / 'train_data_marginal_dists.pt'
        n_atoms_hist_file = Path(__file__).parent.parent / 'data' / 'qm9' / 'train_data_n_atoms_histogram.pt'
        load_kwargs = {'marginal_dists_file': marginal_dists_file, 'n_atoms_hist_file': n_atoms_hist_file}
    else:
        load_kwargs = {}
    
    ckpt_path = model_dir / 'checkpoints' / 'last.ckpt'
    model = FlowMol.load_from_checkpoint(ckpt_path, **load_kwargs)

    if model_name == 'geom_ctmc':
        # temporary fix - this model was trained with certain
        # parameters for sampling that were later found to be suboptimal,
        # so here we set the default sampling parameters
        # to what we've found to give better results 
        model.vector_field.eta = 30.0
        model.vector_field.hc_thresh = 0.9
        

    return model

def download_remote_model_dir(local_model_dir: Path):

    print("Downloading pretrained model...")

    # make local_model_dir an absolute path
    local_model_dir = local_model_dir.resolve()

    # get download location
    local_download_path = local_model_dir.parent
    local_download_path = local_download_path.resolve()

    # get location of remote model dir
    model_name = local_model_dir.name
    remote_model_dir = f"https://bits.csb.pitt.edu/files/FlowMol/trained_models_v02/{model_name}/"

    # download the model
    wget_cmd = f"wget -r -np -nH --cut-dirs=3 --reject 'index.html*' -P {local_download_path} {remote_model_dir}"
    result = subprocess.run(wget_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error downloading model: {result.stderr}")