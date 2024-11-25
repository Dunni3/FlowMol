from pathlib import Path
from flowmol.models.flowmol import FlowMol

def load_model(model_name: str) -> FlowMol:
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

    model_dir = Path(__file__).parent / 'trained_models' / model_name

    if not model_dir.exists():
        supported_models = [p.name for p in model_dir.parent.iterdir() if p.is_dir()]

        if len(supported_models) == 0:
            raise FileNotFoundError("No trained models found. Follow readme instructions to download pre-trained models.")

        supported_models = '\n'.join(supported_models)
        raise FileNotFoundError(f"Model {model_name} not found. Supported models:\n{supported_models}")
    
    ckpt_path = model_dir / 'checkpoints' / 'last.ckpt'
    model = FlowMol.load_from_checkpoint(ckpt_path)

    if model_name == 'geom_ctmc':
        # temporary fix - this model was trained with certain
        # parameters for sampling that were later found to be suboptimal,
        # so here we set the default sampling parameters
        # to what we've found to give better results 
        model.vector_field.eta = 30.0
        model.vector_field.hc_thresh = 0.9

    return model