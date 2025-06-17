import argparse
import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm


def sdf_to_xyz(mol, filename):
    """Convert RDKit mol object to XYZ format for xTB input."""
    with open(filename, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}\n")


def run_xtb_optimization(xyz_filename, output_prefix, charge, work_dir):
    """Run xTB optimization and capture output."""
    output_filename = os.path.join(work_dir, f"{output_prefix}_xtb_output.out")

    # Pass the charge to xTB using --charge flag
    command = f"cd {work_dir} && xtb {os.path.basename(xyz_filename)} --opt --charge {charge} --namespace {output_prefix} > {os.path.basename(output_filename)}"

    subprocess.run(command, shell=True)
    with open(output_filename, 'r') as f:
        xtb_output = f.read()
    return xtb_output


def parse_xtb_output(xtb_output):
    """Parse xTB output to get total energy gain and total RMSD."""
    total_energy_gain = None
    total_rmsd = None

    lines = xtb_output.splitlines()
    for line in lines:
        if "total energy gain" in line:
            total_energy_gain = float(line.split()[6])  # in kcal/mol
        elif "total RMSD" in line:
            total_rmsd = float(line.split()[5])  # in Angstroms

    return total_energy_gain, total_rmsd


def parse_xtbtopo_mol(xtbtopo_filename):
    """Parse xtbtopo.mol file and return an RDKit molecule object."""
    if not os.path.exists(xtbtopo_filename):
        raise FileNotFoundError(f"No such file or directory: '{xtbtopo_filename}'")

    with open(xtbtopo_filename, 'r') as f:
        mol_block = f.read()
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("Failed to create RDKit molecule from xtbtopo.mol")
    return mol


def write_mol_to_sdf(mol, f):
    """Write the RDKit molecule object to an SDF file manually, including custom properties."""
    mol_block = Chem.MolToMolBlock(mol, kekulize=False)
    f.write(mol_block)
    f.write("\n")
    # Write properties
    for prop_name in mol.GetPropNames():
        prop_value = mol.GetProp(prop_name)
        f.write(f">  <{prop_name}>\n{prop_value}\n\n")
    f.write("$$$$\n")


def get_molecule_charge(mol):
    """Calculate the total formal charge of a molecule by summing the formal charges of its atoms."""
    total_charge = 0
    for atom in mol.GetAtoms():
        total_charge += atom.GetFormalCharge()
    return total_charge


def process_molecule(args, temp_dir):
    """Process a single molecule: run xTB optimization, parse output, and return the optimized molecule."""
    i, mol = args
    if mol is None:
        return None, None, None

    # Create paths within the temp directory
    xyz_filename = os.path.join(temp_dir, f"mol_{i}.xyz")
    output_prefix = f"mol_{i}"
    xtb_topo_filename = os.path.join(temp_dir, f"{output_prefix}.xtbtopo.mol")

    sdf_to_xyz(mol, xyz_filename)

    # Get the formal charge of the molecule
    charge = get_molecule_charge(mol)

    try:
        # Pass the charge to the xTB optimization
        xtb_output = run_xtb_optimization(xyz_filename, output_prefix, charge, temp_dir)
        total_energy_gain, total_rmsd = parse_xtb_output(xtb_output)

        if not os.path.exists(xtb_topo_filename):
            raise FileNotFoundError(f"Expected xtbtopo.mol file not found: '{xtb_topo_filename}'")

        optimized_mol = parse_xtbtopo_mol(xtb_topo_filename)
        return optimized_mol, total_energy_gain, total_rmsd

    except Exception as e:
        print(f"Error processing molecule {i}: {e}")
        return None, None, None


def write_results_to_file(output_sdf, optimized_mols, how="w"):
    """Write the optimized molecules to the output SDF file."""
    with open(output_sdf, how) as f:
        for mol in optimized_mols:
            write_mol_to_sdf(mol, f)


def main_fn(input_sdf, output_sdf, init_sdf):
    suppl = Chem.SDMolSupplier(input_sdf, sanitize=False, removeHs=False)
    optimized_mols = []
    init_mols = []

    # Remove existing output file to start fresh
    if os.path.exists(output_sdf):
        os.remove(output_sdf)
    
    # Create a temporary directory for all intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            for task in tqdm(enumerate(suppl)):
                optimized_mol, total_energy_gain, total_rmsd = process_molecule(task, temp_dir)
                if optimized_mol is not None:
                    if task[1].HasProp("_Name"):
                        optimized_mol.SetProp("_Name", task[1].GetProp("_Name"))
                    else:
                        optimized_mol.SetProp("_Name", str(task[0]))
                    if total_energy_gain is not None:
                        optimized_mol.SetProp("energy_gain", f"{total_energy_gain:.4f}")
                    if total_rmsd is not None:
                        optimized_mol.SetProp("RMSD", f"{total_rmsd:.4f}")
                    init_mols.append(task[1])
                    optimized_mols.append(optimized_mol)
        finally:
            # Write remaining optimized molecules to the SDF file
            if optimized_mols:
                write_results_to_file(output_sdf, optimized_mols)
                write_results_to_file(init_sdf, init_mols)

    print(f"Successfully processed molecules.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_sdf", type=str, required=True, help="Path to input .sdf file")
    parser.add_argument("--output_sdf", type=str, required=True, help="Path to output optimized .sdf")
    parser.add_argument("--init_sdf", type=str, required=True, help="Path to output initial structures .sdf")
    args = parser.parse_args()
    main_fn(args.input_sdf, args.output_sdf, args.init_sdf)
