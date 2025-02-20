import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from multiprocessing import Pool
from flowmol.model_utils import load as load_fm
from flowmol.data_processing.utils import get_upper_edge_mask
from flowmol.analysis.molecule_builder import SampledMolecule
import pickle
from rdkit.Chem import AllChem as Chem

def parse_args():
    p = argparse.ArgumentParser(description="Get data valencies")
    p.add_argument('--config', type=Path, required=True, help='Input file path')
    p.add_argument('--split', type=str, default='val', help='Split to use for data')
    p.add_argument('--output', type=Path, required=True, help='Output file path')
    p.add_argument('--n_mols', type=int, default=None)
    p.add_argument('--n_cpus', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=100) 
    
    args = p.parse_args()

    return args

def graph_to_sampled_mol(g, atom_type_map):
    for feat in 'xace':
        if feat == 'e':
            data_src = g.edata
        else:
            data_src = g.ndata
        data_src[f'{feat}_1'] = data_src[f'{feat}_1_true']

    g.edata['ue_mask'] = get_upper_edge_mask(g)
    sampled_mol = SampledMolecule(g, atom_type_map, ctmc_mol=True)

    return sampled_mol

def process_batch(graphs, atom_type_map, batch_idx, n_batches):

    odd_balls = set([
        ('C', 0, 5),
        ('C', 0, 2),
        ('N', 0, 4),
        ('H', 1, 0),
        ('O', 0, 3)
    ])
    odd_ball_dir = Path('/home/ian/projects/mol_diffusion/mol-fm/local/valencies/oddballs')


    valency_table = {}
    for gidx, g in enumerate(graphs):
        sampled_mol = graph_to_sampled_mol(g, atom_type_map)
        atom_types, atom_charges, valencies = sampled_mol.atom_types, sampled_mol.atom_charges, sampled_mol.valencies
        atom_charges = atom_charges.tolist()
        valencies = valencies.tolist()

        oddballs_found = []

        for atom_type, atom_charge, atom_valence in zip(atom_types, atom_charges, valencies):

            if (atom_type, atom_charge, atom_valence) in odd_balls:

                oddballs_found.append((atom_type, atom_charge, atom_valence))

            if atom_type not in valency_table:
                valency_table[atom_type] = {}

            type_table = valency_table[atom_type]

            if atom_charge not in type_table:
                type_table[atom_charge] = []
            charge_table = type_table[atom_charge]

            if atom_valence not in charge_table:
                charge_table.append(atom_valence)

        if len(oddballs_found) > 0:

            # convert oddballs found to a string
            oddballs_found = set(oddballs_found)
            oddballs_found = [f'{atom_type}{atom_charge}{atom_valence}' for atom_type, atom_charge, atom_valence in oddballs_found]
            oddball_str = '_'.join(oddballs_found)

            rdmol = sampled_mol.rdkit_mol
            output_file = odd_ball_dir / f'{oddball_str}_g{gidx}_b{batch_idx}.sdf'
            Chem.MolToMolFile(rdmol, str(output_file))


    print(f'Batch {batch_idx+1}/{n_batches} processed', flush=True)

    return valency_table

def merge_valency_tables(tables: List[Dict]) -> Dict:
    merged_table = {}
    for table in tables:
        for atom_type, charge_table in table.items():
            if atom_type not in merged_table:
                merged_table[atom_type] = {}
            for charge, valencies in charge_table.items():
                if charge not in merged_table[atom_type]:
                    merged_table[atom_type][charge] = []
                for valence in valencies:
                    if valence not in merged_table[atom_type][charge]:
                        merged_table[atom_type][charge].append(valence)
    return merged_table

def batch_generator(dataset, batch_size, n_mols):

    if n_mols is not None:
        max_n_mols = n_mols
    else:
        max_n_mols = len(dataset)+1

    for i in range(0, len(dataset), batch_size):
        start_idx = i
        end_idx = min(i + batch_size, len(dataset))

        if end_idx > max_n_mols:
            break

        batch = [ dataset[i] for i in range(start_idx, end_idx) ]

        yield batch

def run_single_process(dataset, atom_type_map):

    n_batches = len(dataset) // args.batch_size


    for batch_idx, batch in enumerate(batch_generator(dataset, args.batch_size, args.n_mols)):
        valency_table = process_batch(batch, atom_type_map, batch_idx, n_batches)
        valency_tables.append(valency_table)

    valency_table = merge_valency_tables(valency_tables)
    return valency_table

def run_multi_process(dataset, atom_type_map, n_cpus):

    n_batches = len(dataset) // args.batch_size

    with Pool(args.n_cpus) as pool:

        for batch_idx, batch in enumerate(batch_generator(dataset, args.batch_size, args.n_mols)):
            pool.apply_async(process_batch, args=(batch, atom_type_map, batch_idx, n_batches), callback=valency_tables.append)

        pool.close()
        pool.join()

    valency_table = merge_valency_tables(valency_tables)
    return valency_table


valency_tables = []

if __name__ == "__main__":
    args = parse_args()

    config = load_fm.read_config_file(args.config)

    # manually set processed data dir
    config['dataset']['processed_data_dir'] = '/home/ian/projects/mol_diffusion/mol-fm/data/geom5_noarom'


    data_module = load_fm.data_module_from_config(config)
    dataset = data_module.load_dataset(args.split)
    atom_type_map = data_module.dataset_config['atom_map']


    if args.n_cpus == 1:
        valency_table = run_single_process(dataset, atom_type_map)
    else:
        valency_table = run_multi_process(dataset, atom_type_map, args.n_cpus)

    # Save valency table to a pickle file
    with open(args.output, 'wb') as f:
        pickle.dump(valency_table, f)

    # Save valency table to a plain text file
    txt_output = args.output.with_suffix('.txt')
    with open(txt_output, 'w') as f:
        for atom_type, charge_table in valency_table.items():
            f.write(f'Atom type: {atom_type}\n')
            for charge, valencies in charge_table.items():
                f.write(f'  Charge: {charge}\n')
                f.write(f'    Valences: {", ".join(map(str, valencies))}\n')

