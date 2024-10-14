import csv
from networkx.algorithms.components import node_connected_component
import networkx as nx
import numpy as np
from rdkit.Chem.BRICS import BRICSDecompose, BreakBRICSBonds, FindBRICSBonds
from torch.utils.data import Dataset, DataLoader
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data, Batch
from copy import deepcopy
import math
import random
import yaml
from rdkit.Chem import Recap
from model.preprocess import GetCCSingleBonds, fragment_generator

torch.multiprocessing.set_sharing_strategy('file_system')

ATOM_LIST = list(range(0, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def get_fragment_indices(mol):
    bonds = mol.GetBonds()
    edges = []
    for bond in bonds:
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    molGraph = nx.Graph(edges)

    BRICS_bonds = list(FindBRICSBonds(mol))
    break_bonds = [b[0] for b in BRICS_bonds]
    break_atoms = [b[0][0] for b in BRICS_bonds] + [b[0][1] for b in BRICS_bonds]
    molGraph.remove_edges_from(break_bonds)

    indices = []
    for atom in break_atoms:
        n = node_connected_component(molGraph, atom)
        if len(n) > 3 and n not in indices:
            indices.append(n)
    indices = set(map(tuple, indices))

    return indices


def get_frags(mol):
    ref_indices = get_fragment_indices(mol)
    frags = list(BRICSDecompose(mol, returnMols=True))
    mol2 = BreakBRICSBonds(mol)
    extra_indices = []
    for i, atom in enumerate(mol2.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            extra_indices.append(i)
    extra_indices = set(extra_indices)
    frag_mols = []
    frag_indices = []
    for frag in frags:
        indices = mol2.GetSubstructMatches(frag)
        if len(indices) == 1:
            idx = indices[0]
            idx = set(idx) - extra_indices
            if len(idx) > 3:
                frag_mols.append(frag)
                frag_indices.append(idx)
        else:
            for idx in indices:
                idx = set(idx) - extra_indices
                if len(idx) > 3:
                    for ref_idx in ref_indices:
                        if (tuple(idx) == ref_idx) and (idx not in frag_indices):
                            frag_mols.append(frag)
                            frag_indices.append(idx)

    return frag_mols, frag_indices


def get_ccsingle_indices(mol, frags):
    bonds = mol.GetBonds()
    edges = []
    for bond in bonds:
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    molGraph = nx.Graph(edges)

    broken_bonds = []
    mol2 = Chem.RWMol(mol)
    for frag in frags:
        frag_mol_demo = Chem.MolFromSmiles(frag)
        frag_mol = Chem.DeleteSubstructs(frag_mol_demo, Chem.MolFromSmiles('[*]'))
        matches = mol2.GetSubstructMatches(frag_mol)
        for match in matches:
            for bond_idx in frag_mol.GetBonds():
                bond_atoms = bond_idx.GetBeginAtomIdx(), bond_idx.GetEndAtomIdx()
                bond_atoms = tuple(match[i] for i in bond_atoms)
                if bond_atoms not in broken_bonds:
                    broken_bonds.append(bond_atoms)
    break_atoms = [b[0] for b in broken_bonds] + [b[1] for b in broken_bonds]
    molGraph.remove_edges_from(broken_bonds)

    indices = []

    for atom in break_atoms:
        n = node_connected_component(molGraph, atom)
        indices.append(n)
    indices = set(map(tuple, indices))

    return indices


def get_ccsinfle_frags(mol):
    bonds = GetCCSingleBonds(mol)
    bonds_id = [item[0] for item in bonds]
    frag_i = fragment_generator(mol, bonds_id)
    ref_indices = get_ccsingle_indices(mol, frag_i)
    frags = list(BRICSDecompose(mol, returnMols=True))
    mol2 = BreakBRICSBonds(mol)
    extra_indices = []
    for i, atom in enumerate(mol2.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            extra_indices.append(i)
    extra_indices = set(extra_indices)
    frag_mols = []
    frag_indices = []
    for frag in frags:
        indices = mol2.GetSubstructMatches(frag)
        if len(indices) == 1:
            idx = indices[0]
            idx = set(idx) - extra_indices
            if len(idx) > 3:
                frag_mols.append(frag)
                frag_indices.append(idx)
        else:
            for idx in indices:
                idx = set(idx) - extra_indices
                if len(idx) > 3:
                    for ref_idx in ref_indices:
                        if (tuple(idx) == ref_idx) and (idx not in frag_indices):
                            frag_mols.append(frag)
                            frag_indices.append(idx)

    return frag_mols, frag_indices


def get_recap_indices(mol):
    bonds = mol.GetBonds()
    edges = []
    for bond in bonds:
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    molGraph = nx.Graph(edges)

    hierarch = Recap.RecapDecompose(mol)
    frags = hierarch.GetLeaves().keys()

    broken_bonds = []
    mol2 = Chem.RWMol(mol)
    for frag in frags:
        frag_mol_demo = Chem.MolFromSmiles(frag)
        frag_mol = Chem.DeleteSubstructs(frag_mol_demo, Chem.MolFromSmiles('[*]'))
        matches = mol2.GetSubstructMatches(frag_mol)
        for match in matches:
            for bond_idx in frag_mol.GetBonds():
                bond_atoms = bond_idx.GetBeginAtomIdx(), bond_idx.GetEndAtomIdx()
                bond_atoms = tuple(match[i] for i in bond_atoms)
                if bond_atoms not in broken_bonds:
                    broken_bonds.append(bond_atoms)
    break_atoms = [b[0] for b in broken_bonds] + [b[1] for b in broken_bonds]
    molGraph.remove_edges_from(broken_bonds)

    indices = []


    indices = set(map(tuple, indices))

    return indices


def get_recap_frags(mol):
    ref_indices = get_recap_indices(mol)
    frags = list(BRICSDecompose(mol, returnMols=True))
    mol2 = BreakBRICSBonds(mol)
    extra_indices = []
    for i, atom in enumerate(mol2.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            extra_indices.append(i)
    extra_indices = set(extra_indices)
    frag_mols = []
    frag_indices = []
    for frag in frags:
        indices = mol2.GetSubstructMatches(frag)
        if len(indices) == 1:
            idx = indices[0]
            idx = set(idx) - extra_indices
            if len(idx) > 3:
                frag_mols.append(frag)
                frag_indices.append(idx)
        else:
            for idx in indices:
                idx = set(idx) - extra_indices
                if len(idx) > 3:
                    for ref_idx in ref_indices:
                        if (tuple(idx) == ref_idx) and (idx not in frag_indices):
                            frag_mols.append(frag)
                            frag_indices.append(idx)

    return frag_mols, frag_indices


def get_data_mol(smiles_list):
    lis = []
    for n in smiles_list:
        mol = Chem.MolFromSmiles(n[0])
        lis.append(mol)

    return lis


class dataprocess_brics(Dataset):
    def __init__(self, smiles_data):
        super(dataprocess_brics, self).__init__()
        self.smiles_data = smiles_data

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index][0])
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        row, col, edge_feat = [], [], []

        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

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

        # mask the subgraph of molecule
        num_mask_nodes = max([1, math.floor(0.25 * N)])
        num_mask_edges = max([0, math.floor(0.25 * M)])
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
        mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_i = [2 * i for i in mask_edges_i_single] + [2 * i + 1 for i in mask_edges_i_single]
        mask_edges_j = [2 * i for i in mask_edges_j_single] + [2 * i + 1 for i in mask_edges_j_single]

        x_i = deepcopy(x)

        for atom_idx in mask_nodes_i:
            x_i[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_i = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2 * (M - num_mask_edges), 2), dtype=torch.long)

        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:, count] = edge_index[:, bond_idx]
                edge_attr_i[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(x)
        for atom_idx in mask_nodes_j:
            x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_j = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2 * (M - num_mask_edges), 2), dtype=torch.long)

        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_j:
                edge_index_j[:, count] = edge_index[:, bond_idx]
                edge_attr_j[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)

        frag_mols, frag_indices = get_frags(mol)

        return data_i, data_j, mol, N, frag_mols, frag_indices

    def __len__(self):

        return len(self.smiles_data)


class dataprocess_recap(Dataset):
    def __init__(self, smiles_data):
        super(dataprocess_recap, self).__init__()
        self.smiles_data = smiles_data

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index][0])
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        row, col, edge_feat = [], [], []

        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

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

        # mask the subgraph of molecule
        num_mask_nodes = max([1, math.floor(0.25 * N)])
        num_mask_edges = max([0, math.floor(0.25 * M)])
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
        mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_i = [2 * i for i in mask_edges_i_single] + [2 * i + 1 for i in mask_edges_i_single]
        mask_edges_j = [2 * i for i in mask_edges_j_single] + [2 * i + 1 for i in mask_edges_j_single]

        x_i = deepcopy(x)

        for atom_idx in mask_nodes_i:
            x_i[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_i = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2 * (M - num_mask_edges), 2), dtype=torch.long)

        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:, count] = edge_index[:, bond_idx]
                edge_attr_i[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(x)
        for atom_idx in mask_nodes_j:
            x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_j = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2 * (M - num_mask_edges), 2), dtype=torch.long)

        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_j:
                edge_index_j[:, count] = edge_index[:, bond_idx]
                edge_attr_j[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)

        frag_mols, frag_indices = get_recap_frags(mol)

        return data_i, data_j, mol, N, frag_mols, frag_indices

    def __len__(self):

        return len(self.smiles_data)


class dataprocess_ccsingle(Dataset):
    def __init__(self, smiles_data):
        super(dataprocess_ccsingle, self).__init__()
        self.smiles_data = smiles_data

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index][0])
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        row, col, edge_feat = [], [], []

        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

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

        # mask the subgraph of molecule
        num_mask_nodes = max([1, math.floor(0.25 * N)])
        num_mask_edges = max([0, math.floor(0.25 * M)])
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
        mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_i = [2 * i for i in mask_edges_i_single] + [2 * i + 1 for i in mask_edges_i_single]
        mask_edges_j = [2 * i for i in mask_edges_j_single] + [2 * i + 1 for i in mask_edges_j_single]

        x_i = deepcopy(x)

        for atom_idx in mask_nodes_i:
            x_i[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_i = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2 * (M - num_mask_edges), 2), dtype=torch.long)

        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:, count] = edge_index[:, bond_idx]
                edge_attr_i[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(x)
        for atom_idx in mask_nodes_j:
            x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
        edge_index_j = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2 * (M - num_mask_edges), 2), dtype=torch.long)

        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_j:
                edge_index_j[:, count] = edge_index[:, bond_idx]
                edge_attr_j[count, :] = edge_attr[bond_idx, :]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)

        frag_mols, frag_indices = get_ccsinfle_frags(mol)

        return data_i, data_j, mol, N, frag_mols, frag_indices

    def __len__(self):

        return len(self.smiles_data)


def attr_mask(data, aug_ratio=0.25):
    data_i = deepcopy(data)
    data_j = deepcopy(data)
    node_num, _ = data_i.x.size()
    _, edge_num = data_j.edge_index.size()
    mask_num_i = int(node_num * aug_ratio)
    mask_num_j = int(edge_num * aug_ratio)
    token_i = data_i.x.float().mean(dim=0)
    idx_mask_i = np.random.choice(node_num, mask_num_i, replace=False)
    idx_mask_j = np.random.choice(edge_num, min(mask_num_j, edge_num), replace=False)
    data_i.x[idx_mask_i] = token_i
    data_j.edge_index = data_j.edge_index[:, idx_mask_j]
    return data_i, data_j, idx_mask_j


class attrprocess_brics(Dataset):
    def __init__(self, smiles_data):
        super(attrprocess_brics, self).__init__()
        self.smiles_data = smiles_data

    def __getitem__(self, i):

        # 特征掩蔽
        atom_features = []
        bond_indices = []
        bond_types = []

        mol = Chem.MolFromSmiles(self.smiles_data[i][0])
        for atom in mol.GetAtoms():
            atom_features.append(atom.GetAtomicNum())

        for bond in mol.GetBonds():
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()

            bond_indices.append([begin_atom_idx, end_atom_idx])
            bond_types.append(bond_type)

        typ_idx = []
        chiral_idx = []
        ato_number = []
        rows, cols, edge_feats = [], [], []

        for atom in mol.GetAtoms():
            typ_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chiral_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            ato_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(typ_idx, dtype=torch.float).view(-1, 1)
        x2 = torch.tensor(chiral_idx, dtype=torch.float).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [start, end]
            cols += [end, start]
            edge_feats.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feats.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr_demo = torch.tensor(np.array(edge_feats), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_demo)

        fragments_i, fragments_j, idx = attr_mask(data)
        frag_mols, frag_indices = get_frags(mol)
        N = mol.GetNumAtoms()

        return fragments_i, fragments_j, mol, N, frag_mols, frag_indices

    def __len__(self):

        return len(self.smiles_data)


class attrprocess_recap(Dataset):
    def __init__(self, smiles_data):
        super(attrprocess_recap, self).__init__()
        self.smiles_data = smiles_data

    def __getitem__(self, i):

        atom_features = []
        bond_indices = []
        bond_types = []

        mol = Chem.MolFromSmiles(self.smiles_data[i][0])
        for atom in mol.GetAtoms():
            atom_features.append(atom.GetAtomicNum())

        for bond in mol.GetBonds():
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()

            bond_indices.append([begin_atom_idx, end_atom_idx])
            bond_types.append(bond_type)

        typ_idx = []
        chiral_idx = []
        ato_number = []
        rows, cols, edge_feats = [], [], []

        for atom in mol.GetAtoms():
            typ_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chiral_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            ato_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(typ_idx, dtype=torch.float).view(-1, 1)
        x2 = torch.tensor(chiral_idx, dtype=torch.float).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [start, end]
            cols += [end, start]
            edge_feats.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feats.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr_demo = torch.tensor(np.array(edge_feats), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_demo)

        fragments_i, fragments_j, idx = attr_mask(data)
        frag_mols, frag_indices = get_recap_frags(mol)
        N = mol.GetNumAtoms()

        return fragments_i, fragments_j, mol, N, frag_mols, frag_indices

    def __len__(self):
        return len(self.smiles_data)


class attrprocess_ccsingle(Dataset):
    def __init__(self, smiles_data):
        super(attrprocess_ccsingle, self).__init__()
        self.smiles_data = smiles_data

    def __getitem__(self, i):

        atom_features = []
        bond_indices = []
        bond_types = []

        mol = Chem.MolFromSmiles(self.smiles_data[i][0])
        for atom in mol.GetAtoms():
            atom_features.append(atom.GetAtomicNum())

        for bond in mol.GetBonds():
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()

            bond_indices.append([begin_atom_idx, end_atom_idx])
            bond_types.append(bond_type)

        typ_idx = []
        chiral_idx = []
        ato_number = []
        rows, cols, edge_feats = [], [], []

        for atom in mol.GetAtoms():
            typ_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chiral_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            ato_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(typ_idx, dtype=torch.float).view(-1, 1)
        x2 = torch.tensor(chiral_idx, dtype=torch.float).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [start, end]
            cols += [end, start]
            edge_feats.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feats.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr_demo = torch.tensor(np.array(edge_feats), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_demo)

        fragments_i, fragments_j, idx = attr_mask(data)
        frag_mols, frag_indices = get_ccsinfle_frags(mol)
        N = mol.GetNumAtoms()

        return fragments_i, fragments_j, mol, N, frag_mols, frag_indices

    def __len__(self):

        return len(self.smiles_data)


def pre_pro_brics(smiles_data):
    pro1_brics = dataprocess_brics(smiles_data)
    pro2_brics = attrprocess_brics(smiles_data)

    return pro1_brics, pro2_brics


def pre_pro_recap(smiles_data):
    pro1_recap = dataprocess_recap(smiles_data)
    pro2_recap = attrprocess_recap(smiles_data)

    return pro1_recap, pro2_recap


def pre_pro_ccsingle(smiles_data):
    pro1_ccsingle = dataprocess_ccsingle(smiles_data)
    pro2_ccsingle = attrprocess_ccsingle(smiles_data)

    return pro1_ccsingle, pro2_ccsingle


def collate_fn1(batch):
    gis, gjs, mols, atom_nums, frag_mols, frag_indices = zip(*batch)
    frag_mols = [j for i in frag_mols for j in i]
    gis = Batch.from_data_list(gis)
    gjs = Batch.from_data_list(gjs)

    gis.motif_batch = torch.zeros(gis.x.size(0), dtype=torch.long)
    gjs.motif_batch = torch.zeros(gjs.x.size(0), dtype=torch.long)

    curr_indicator = 1
    curr_num = 0
    for N, indices in zip(atom_nums, frag_indices):
        for idx in indices:
            curr_idx = np.array(list(idx)) + curr_num
            gis.motif_batch[curr_idx] = curr_indicator
            gjs.motif_batch[curr_idx] = curr_indicator
            curr_indicator += 1
        curr_num += N

    return gis, gjs, mols, frag_mols


def read_mols(path):
    smiles_data = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for s in csv_reader:
            smiles_data.append(s)

    return smiles_data


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, feat_dim):
        super(object, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.task = task
        self.valid_size = valid_size

    def get_data_loaders(self):
        mol_path = '../.csv'
        mol_data = read_mols(mol_path)
        print('预训练数据量:', len(mol_data))

        # data split
        num_train = len(mol_data)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_smiles = [mol_data[i] for i in train_idx]
        valid_smiles = [mol_data[i] for i in valid_idx]

        del mol_data

        pro1_brics_train, pro2_brics_train = pre_pro_brics(train_smiles)
        pro1_brics_valid, pro2_brics_valid = pre_pro_brics(valid_smiles)

        train_loader_b1 = DataLoader(pro1_brics_train, batch_size=self.batch_size, collate_fn=collate_fn1,
                                     num_workers=self.num_workers, drop_last=True, shuffle=True
                                     )
        train_loader_b2 = DataLoader(pro2_brics_train, batch_size=self.batch_size, collate_fn=collate_fn1,
                                     num_workers=self.num_workers, drop_last=True, shuffle=True
                                     )
        valid_loader_b1 = DataLoader(pro1_brics_valid, batch_size=self.batch_size, collate_fn=collate_fn1,
                                     num_workers=self.num_workers, drop_last=True, shuffle=True
                                     )
        valid_loader_b2 = DataLoader(pro2_brics_valid, batch_size=self.batch_size, collate_fn=collate_fn1,
                                     num_workers=self.num_workers, drop_last=True, shuffle=True
                                     )

        del pro1_brics_train, pro2_brics_train, pro1_brics_valid, pro2_brics_valid

        pro1_recap_train, pro2_recap_train = pre_pro_recap(train_smiles)
        pro1_recap_valid, pro2_recap_valid = pre_pro_recap(valid_smiles)

        del train_smiles, valid_smiles

        train_loader_r1 = DataLoader(pro1_recap_train, batch_size=self.batch_size, collate_fn=collate_fn1,
                                     num_workers=self.num_workers, drop_last=True, shuffle=True
                                     )
        train_loader_r2 = DataLoader(pro2_recap_train, batch_size=self.batch_size, collate_fn=collate_fn1,
                                     num_workers=self.num_workers, drop_last=True, shuffle=True
                                     )
        valid_loader_r1 = DataLoader(pro1_recap_valid, batch_size=self.batch_size, collate_fn=collate_fn1,
                                     num_workers=self.num_workers, drop_last=True, shuffle=True
                                     )
        valid_loader_r2 = DataLoader(pro2_recap_valid, batch_size=self.batch_size, collate_fn=collate_fn1,
                                     num_workers=self.num_workers, drop_last=True, shuffle=True
                                     )

        del pro1_recap_train, pro2_recap_train, pro1_recap_valid, pro2_recap_valid

        return (train_loader_b1, train_loader_b2, valid_loader_b1, valid_loader_b2,
                train_loader_r1, train_loader_r2, valid_loader_r1, valid_loader_r2)


if __name__ == "__main__":
    config = yaml.load(open("../config.yaml", "r"),
                       Loader=yaml.FullLoader)
    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    (train_loader_b1, train_loader_b2, valid_loader_b1, valid_loader_b2,
     train_loader_r1, train_loader_r2, valid_loader_r1, valid_loader_r2) = dataset.get_data_loaders()

