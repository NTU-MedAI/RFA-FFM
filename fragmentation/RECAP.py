import csv
import pandas as pd
from rdkit.Chem import Recap
from rdkit import Chem


df = pd.read_csv('/.csv')
smiles_list = df['smiles'].tolist()
f = open('', 'w', encoding='utf-8')
csv_write = csv.writer(f)
for item in smiles_list:
    m = Chem.MolFromSmiles(item)
    a = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    hierarch = Recap.RecapDecompose(m)
    for frags in hierarch.GetLeaves().keys():
        csv_write.writerow([frags])


