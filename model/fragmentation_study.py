import os
from rdkit import Chem
import matplotlib.pyplot as plt


def read_sdf_to_smiles(sdf_file):
    suppl = Chem.SDMolSupplier(sdf_file)
    results = []
    for mol in suppl:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            props = mol.GetPropsAsDict()
            results.append({
                'smiles': smiles,
                **props
            })
    return results


def process_sdf_files_in_folder(folder_path):

    sdf_files = [f for f in os.listdir(folder_path) if f.endswith('.sdf')]
    sdf_files.sort()

    all_results = []
    for sdf_file in sdf_files:
        file_path = os.path.join(folder_path, sdf_file)
        print(f"Processing {file_path}...")
        results = read_sdf_to_smiles(file_path)
        all_results.extend(results)

    return all_results

