import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolTransforms


from geom_utils.utils import generate_canonical_key


def compute_bond_angles_diff(pair):
    init_mol, opt_mol = pair
    bond_angles = {}
    init_conf = init_mol.GetConformer()
    opt_conf = opt_mol.GetConformer()

    for atom in init_mol.GetAtoms():
        neighbors = atom.GetNeighbors()
        if len(neighbors) < 2:
            continue

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                idx1, idx2, idx3 = neighbors[i].GetIdx(), atom.GetIdx(), neighbors[j].GetIdx()
                atom1_type, atom2_type, atom3_type = init_mol.GetAtomWithIdx(
                    idx1).GetAtomicNum(), init_mol.GetAtomWithIdx(
                    idx2).GetAtomicNum(), init_mol.GetAtomWithIdx(idx3).GetAtomicNum()
                bond_type_1 = int(init_mol.GetBondBetweenAtoms(idx1, idx2).GetBondType())
                bond_type_2 = int(init_mol.GetBondBetweenAtoms(idx2, idx3).GetBondType())

                angle_init = rdMolTransforms.GetAngleDeg(init_conf, idx1, idx2, idx3)
                angle_opt = rdMolTransforms.GetAngleDeg(opt_conf, idx1, idx2, idx3)

                diff = min(np.abs(angle_init - angle_opt), 360 - np.abs(angle_init - angle_opt))

                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2,
                                             atom3_type)
                if key not in bond_angles:
                    bond_angles[key] = [[], 0]
                bond_angles[key][0].append(diff)
                bond_angles[key][1] += 1

    return bond_angles


def compute_bond_lengths_diff(pair):
    init_mol, opt_mol = pair
    bond_lengths = {}

    init_conf = init_mol.GetConformer()
    opt_conf = opt_mol.GetConformer()
    for bond in init_mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom1_type, atom2_type = init_mol.GetAtomWithIdx(
            idx1).GetAtomicNum(), init_mol.GetAtomWithIdx(idx2).GetAtomicNum()
        bond_type_numeric = int(bond.GetBondType())
        init_length = rdMolTransforms.GetBondLength(init_conf, idx1, idx2)
        opt_length = rdMolTransforms.GetBondLength(opt_conf, idx1, idx2)
        diff = np.abs(init_length - opt_length)

        key = generate_canonical_key(atom1_type, bond_type_numeric, atom2_type)
        if key not in bond_lengths:
            bond_lengths[key] = [[], 0]
        bond_lengths[key][0].append(diff)
        bond_lengths[key][1] += 1
    return bond_lengths


def compute_torsion_angles_diff(pair):
    init_mol, opt_mol = pair
    torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    torsion_query = Chem.MolFromSmarts(torsionSmarts)

    torsion_angles = {}

    init_conf = init_mol.GetConformer()
    opt_conf = opt_mol.GetConformer()

    torsion_matches = init_mol.GetSubstructMatches(torsion_query)

    for match in torsion_matches:
        idx2, idx3 = match[0], match[1]
        bond = init_mol.GetBondBetweenAtoms(idx2, idx3)

        for b1 in init_mol.GetAtomWithIdx(idx2).GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in init_mol.GetAtomWithIdx(idx3).GetBonds():
                if b2.GetIdx() == bond.GetIdx() or b2.GetIdx() == b1.GetIdx():
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                if idx4 == idx1:
                    continue

                atom1_type, atom2_type, atom3_type, atom4_type = init_mol.GetAtomWithIdx(
                    idx1).GetAtomicNum(), init_mol.GetAtomWithIdx(
                    idx2).GetAtomicNum(), init_mol.GetAtomWithIdx(
                    idx3).GetAtomicNum(), init_mol.GetAtomWithIdx(idx4).GetAtomicNum()
                bond_type_1 = int(b1.GetBondType())
                bond_type_2 = int(bond.GetBondType())
                bond_type_3 = int(b2.GetBondType())

                init_angle = rdMolTransforms.GetDihedralDeg(init_conf, idx1, idx2, idx3, idx4)
                opt_angle = rdMolTransforms.GetDihedralDeg(opt_conf, idx1, idx2, idx3, idx4)
                diff = min(np.abs(init_angle - opt_angle), 360 - np.abs(init_angle - opt_angle))
                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2,
                                             atom3_type, bond_type_3, atom4_type)

                if key not in torsion_angles:
                    torsion_angles[key] = [[], 0]
                torsion_angles[key][0].append(diff)
                torsion_angles[key][1] += 1

    return torsion_angles
