import csv
import pandas as pd
from rdkit.Chem import BRICS
from rdkit import Chem


df = pd.read_csv('/.csv')
smiles_list = df['smiles'].tolist()
f = open('', 'w', encoding='utf-8')
csv_write = csv.writer(f)
for item in smiles_list:
    m = Chem.MolFromSmiles(item)
    mol = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    fragments = BRICS.BRICSDecompose(m)


def get_brics_frags(smiles):
    mol = Chem.MolFromSmiles(smiles)
    frag = BRICS.BRICSDecompose(mol)

    return frag