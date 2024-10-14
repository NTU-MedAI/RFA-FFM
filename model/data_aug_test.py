import csv
import random
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path, target, task):
    print('data_path:', data_path)
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smile']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print('Number of data:', len(smiles_data))
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task='classification'):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task
        self.target = target
        self._indices = None
        self.transform = None

    def get(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1, -1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index], dtype=torch.float).view(1, -1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

        return data

    def len(self):

        return len(list(self.smiles_data))


class MolTestDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size, data_path, target, task, seed=2024):
        super(object, self).__init__()
        self.seed = seed
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task

    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)

        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        validation_split = 0.1
        test_split = 0.1

        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))

        random.shuffle(indices)

        valid_split_point = int((1 - validation_split - test_split) * dataset_size)
        test_split_point = int((1 - test_split) * dataset_size)

        train_idx = indices[:valid_split_point]
        valid_idx = indices[valid_split_point:test_split_point]
        test_idx = indices[test_split_point:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=False, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False)

        test_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=test_sampler,
                                 num_workers=self.num_workers, drop_last=False)

        return train_loader, valid_loader, test_loader
