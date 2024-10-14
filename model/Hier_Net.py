import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Parameter, Bilinear
from torch_scatter import scatter
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.nn.pool.pool import pool_batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from yacs.config import CfgNode as CN
from config import _C
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FeatureAttention(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.mlp = Sequential(
            Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            Linear(channels // reduction, channels, bias=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, batch):
        batch = batch.to(x.device)
        max_result = scatter(x, batch, dim=0, dim_size=batch.max().item() + 1, reduce='max')
        sum_result = scatter(x, batch, dim=0, dim_size=batch.max().item() + 1, reduce='sum')
        max_out = self.mlp(max_result)
        sum_out = self.mlp(sum_result)
        y = torch.relu(max_out + sum_out)
        y = y[batch]

        return x * y


class NTNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, slices, dropout, edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(NTNConv, self).__init__(node_dim=1, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.slices = slices
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.weight_node = Parameter(torch.Tensor(in_channels, out_channels))
        if edge_dim is not None:
            self.weight_edge = Parameter(torch.Tensor(edge_dim,
                                                      out_channels))
        else:
            self.weight_edge = self.register_parameter('weight_edge', None)

        self.bilinear = Bilinear(out_channels, out_channels, slices, bias=False)

        if self.edge_dim is not None:
            self.linear = Linear(3 * out_channels, slices)
        else:
            self.linear = Linear(2 * out_channels, slices)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_node)
        glorot(self.weight_edge)
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):

        x = torch.matmul(x, self.weight_node)

        if self.weight_edge is not None:
            assert edge_attr is not None
            edge_attr = torch.matmul(edge_attr, self.weight_node)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_attr):
        score = self.bilinear(x_i, x_j)

        linear_transform = nn.Linear(edge_attr.size(1), x_i.size(0))
        edge_attr = linear_transform(edge_attr).transpose(0, 1)

        if edge_attr is not None:
            vec = torch.cat((x_i, edge_attr, x_j), 1)
            block_score = self.linear(vec)
        else:
            vec = torch.cat((x_i, x_j), 1)
            block_score = self.linear(vec)

        scores = score + block_score
        alpha = torch.tanh(scores)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        dim_split = self.out_channels // self.slices
        out = torch.max(x_j, edge_attr).view(-1, self.slices, dim_split)

        out = out * alpha.view(-1, self.slices, 1)
        out = out.view(-1, self.out_channels)
        return out

    def __repr__(self):
        return '{}({}, {}, slices={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.slices)


def build_model():
    model = HierNet(in_channels=128,
                    hidden_channels=64,
                    out_channels=_C.MODEL.OUT_DIM,
                    edge_dim=10,
                    num_layers=_C.MODEL.DEPTH,
                    dropout=_C.MODEL.DROPOUT,
                    slices=_C.MODEL.SLICES,
                    f_att=_C.MODEL.F_ATT,
                    r=_C.MODEL.R,
                    brics=_C.MODEL.BRICS,
                    macfrag=_C.MODEL.MacFrag,
                    cl=_C.LOSS.CL_LOSS, )

    return model


def get_device():
    device = 'cuda:0'
    return device


class HierNet(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 slices, dropout, edge_dim, f_att=False, r=4, brics=True, macfrag=True, cl=False, pooling='mean'):
        super(HierNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.f_att = f_att
        self.brics = brics
        self.MacFrag = macfrag
        self.cl = cl
        self.lin_a = Linear(in_channels, hidden_channels)
        self.lin_b = Linear(edge_dim, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.in_channels = in_channels
        self.edge_dim = edge_dim

        if pooling == 'mean':
            self.pool = global_mean_pool

        self.atom_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = NTNConv(hidden_channels, hidden_channels, slices=slices,
                           dropout=dropout, edge_dim=10)
            self.atom_convs.append(conv)

        self.lin_gate = Linear(3 * hidden_channels, hidden_channels)

        if self.f_att:
            self.feature_att = FeatureAttention(channels=hidden_channels, reduction=r)

        if self.brics:
            # mol-frag attention
            self.cross_att = GATConv(hidden_channels, hidden_channels, heads=4,
                                     dropout=dropout, add_self_loops=False,
                                     negative_slope=0.01, concat=False)

        if self.MacFrag:
            # brics-MacFrag attention
            self.cross_att = GATConv(hidden_channels, hidden_channels, heads=8,
                                     dropout=dropout, add_self_loops=False,
                                     negative_slope=0.01, concat=False)

        if self.brics:
            self.out = Linear(2 * hidden_channels, out_channels)
        else:
            self.out = Linear(hidden_channels, out_channels)

        if self.MacFrag:
            self.out = Linear(2 * hidden_channels, out_channels)
        else:
            self.out = Linear(hidden_channels, out_channels)

        if self.cl:
            self.lin_project = Linear(hidden_channels, int(hidden_channels / 2))

    def forward(self, data):
        device = get_device()
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        x = self.bn1(self.lin_a(x))
        x = F.relu(x)
        edge_attr = self.bn2(self.lin_b(edge_attr))
        edge_attr = F.relu(edge_attr)

        batch1 = torch.arange(x.size(0)).to(device)

        for i in range(0, self.num_layers):
            if (edge_index > x.size(0)).any:
                continue
            h = F.relu(self.atom_convs[i](x, edge_index, edge_attr))
            beta = self.lin_gate(torch.cat([x, h, x - h], 1)).relu()
            x = beta * x + (1 - beta) * h
            if self.f_att:
                x = self.feature_att(x, batch1)
        mol_vec = global_add_pool(x, batch1).relu_()

        # MacFrag
        if self.MacFrag:
            frag_m_x = data.frag_x.to(device)
            frag_m_edge_index = data.frag_edge_index.to(device)
            frag_m_edge_attr = data.frag_edge_attr.to(device)
            cluster_m = data.cluster_index.to(device)
            frag_m_x = F.relu(self.lin_a(frag_m_x))

            batch2 = (torch.tensor([j for j in range(frag_m_x.size()[0])])).to(device)

            for j in range(0, self.num_layers):
                if (frag_m_edge_index > x.size(0)).any:
                    continue
                frag_h = F.relu(self.atom_convs[j](frag_m_x, frag_m_edge_index, frag_m_edge_attr))
                beta = self.lin_gate(torch.cat([frag_m_x, frag_h, frag_m_x - frag_h], 1)).relu()
                frag_m_x = beta * frag_m_x + (1 - beta) * frag_h

                if self.f_att:
                    frag_m_x = self.feature_att(frag_m_x, cluster_m)

            frag_m_x = global_add_pool(frag_m_x, cluster_m).relu_()

            cluster_m, perm = consecutive_cluster(cluster_m)
            perm = perm.to(device)
            frag_m_batch = pool_batch(perm, batch2)

            # mol - MacFrag
            row = torch.arange(frag_m_batch.size(0), device=device)
            mol_frag_index = torch.stack([row, frag_m_batch], dim=0)
            frag_m_vec = self.cross_att((frag_m_x, mol_vec), mol_frag_index).relu_()

            vectors_concat = list()
            vectors_concat.append(mol_vec)
            vectors_concat.append(frag_m_vec)

            out_m = torch.cat(vectors_concat, 1)

            if self.cl:
                out_m = F.dropout(out_m, p=self.dropout, training=self.training)
                return self.out(out_m), self.lin_project(mol_vec).relu_(), self.lin_project(frag_m_vec).relu_()
            else:
                out_m = F.dropout(out_m, p=self.dropout, training=self.training)
                return self.out(out_m)

        else:
            assert self.cl is False
            out = F.dropout(mol_vec, p=self.dropout, training=self.training)
            return self.out(out)


def get_config():
    _C = CN()
    cfg = _C.clone()

    return cfg

