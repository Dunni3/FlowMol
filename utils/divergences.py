import numpy as np
from scipy.spatial.distance import jensenshannon
from pathlib import Path
from typing import List

def save_reference_dist(bins, p, output_file: Path):
    np.savez(output_file, bins=bins, p=p)

class DivergenceCalculator:

    def __init__(self, reference_dist_file: Path):
        
        data = np.load(reference_dist_file)
        self.bins = data['bins']
        self.p_ref = data['p']

    def js_divergence(self, energies: List[float]) -> float:
        counts, _ = np.histogram(energies, bins=self.bins, density=False)
        p = counts / counts.sum()
        dist = jensenshannon(p, self.p_ref)
        return dist