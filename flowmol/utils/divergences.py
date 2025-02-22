import numpy as np
from scipy.spatial.distance import jensenshannon
from pathlib import Path
from typing import List

def save_reference_dist(bins, p, output_file: Path):
    np.savez(output_file, bins=bins, p=p)

class DivergenceCalculator:

    def __init__(self, reference_dist_file: Path):

        if not reference_dist_file.exists():
            self.no_reference_dist = True
        else:
            self.no_reference_dist = False

            data = np.load(reference_dist_file)
            self.bins = data['bins']
            self.p_ref = data['p']

    def js_divergence(self, energies: List[float]) -> float:

        if self.no_reference_dist:
            raise ValueError('No reference distribution found')

        counts, _ = np.histogram(energies, bins=self.bins, density=False)
        p = counts / counts.sum()
        dist = jensenshannon(p, self.p_ref)
        return dist