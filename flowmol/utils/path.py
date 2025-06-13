from pathlib import Path
 
def flowmol_root() -> str:
    """Returns the root directory of the flowmol package."""
    import flowmol
    fm_path = str(flowmol.__path__[0])
    return str(Path(fm_path).parent)