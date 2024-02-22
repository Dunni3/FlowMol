import torch
from typing import List

def compute_p_c_given_a(atom_charges: torch.Tensor, atom_types: torch.Tensor, atom_type_map: List[str]) -> torch.Tensor:
    """Computes the conditional distribution of charges given atom type, p(c|a)."""
    charge_idx_to_val = torch.arange(-2,4)
    charge_val_to_idx = {int(val): idx for idx, val in enumerate(charge_idx_to_val)}
    
    n_atom_types = len(atom_type_map)
    n_charges = len(charge_idx_to_val)

    # convert atom types from one-hots to indices
    atom_types = atom_types.float().argmax(dim=1)

    # create a tensor to store the conditional distribution of charges given atom type, p(c|a)
    p_c_given_a = torch.zeros(n_atom_types, n_charges, dtype=torch.float32)


    for atom_idx in range(n_atom_types):
        atom_type_mask = atom_types == atom_idx # mask for atoms with the current atom type
        unique_charges, charge_counts = torch.unique(atom_charges[atom_type_mask], return_counts=True)
        for unique_charge, charge_count in zip(unique_charges, charge_counts):
            charge_idx = charge_val_to_idx[int(unique_charge)]
            p_c_given_a[atom_idx, charge_idx] = charge_count

    p_c_given_a = p_c_given_a / p_c_given_a.sum(dim=1, keepdim=True)
    return p_c_given_a