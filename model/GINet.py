import pickle
import shutil
import numpy as np
import torch
import os
from datetime import datetime
torch.autograd.set_detect_anomaly(True)
import torch
import yaml
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from model.NTXENT_global import NTXentLoss
from model.data_aug import MoleculeDatasetWrapper
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.furtherNTXENT import WeightedNTXentLoss
import datetime


torch.multiprocessing.set_sharing_strategy('file_system')
num_atom_type = 119
num_chirality_tag = 3
num_bond_type = 5
num_bond_direction = 3


try:
    from torch.cuda.amp import GradScaler, autocast

    precision_support = True
    print("PyTorch's AMP is supported.")
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    print("Please ensure you have a version of PyTorch that supports automatic mixed precision.")
    precision_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('../config.yaml',
                    os.path.join(model_checkpoints_folder, 'config.yaml'))


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINEConv, self).__init__()
        emb_dim = 200
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )

        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):

        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        if edge_attr.shape[0] > x_j.shape[0]:
            edge_attr = edge_attr[:x_j.shape[0]]
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class gnnet(nn.Module):
    def __init__(self, num_layer=10, emb_dim=256, num_atom_type=500, num_chirality_tag=500, feat_dim=200,
                 dropout=0, pool='mean'):
        super(gnnet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.dropout = dropout
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        self.device = get_device()
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim, aggr="add"))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim // 2)
        )

    def forward(self, data):

        if data.x.size(1) == 2:
            h = self.x_embedding1(data.x[:, 0].long()) + self.x_embedding2(data.x[:, 1].long())

            for layer in range(self.num_layer):
                h = self.gnns[layer](h, data.edge_index, data.edge_attr)
                h = self.batch_norms[layer](h)
                if layer == self.num_layer - 1:
                    h = F.dropout(h, self.dropout, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.dropout, training=self.training)

            h_global = self.pool(h, data.batch)
            h_global = self.feat_lin(h_global)
            out_global = self.out_lin(h_global)

            h_sub = self.pool(h, data.batch)
            h_sub = self.feat_lin(h_sub)
            out_sub = self.out_lin(h_sub)

            return h_global, out_global, out_sub


loss_list = []
loss_attr_list = []
z_global_list, z_rec_global_list = [], []
z_sub_list, z_rec_sub_list = [], []


def elastic_net_reg(model, l1_lambda, l2_lambda):
    l1_reg = torch.tensor(0.).to(get_device())
    l2_reg = torch.tensor(0.).to(get_device())

    for param in model.parameters():
        l1_reg += torch.norm(param, 1)  # L1正则项
        l2_reg += torch.norm(param, 2) ** 2  # L2正则项

    # 返回总正则化损失，乘以超参数λ
    return l1_lambda * l1_reg + l2_lambda * l2_reg


def get_device():
    device = 'cuda:0'
    return device


class fragCLR(object):
    def __init__(self, data_aug, config):
        dir_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.config = config
        self.device = self._get_device()
        self.data_aug = data_aug

    def _get_device(self):
        device = 'cuda:0'
        print("Running on:", device)
        return device

    def train(self):
        model = gnnet(**self.config["model"]).to(self.device)
        (train_loader_b1, train_loader_b2, valid_loader_b1, valid_loader_b2,
         train_loader_r1, train_loader_r2, valid_loader_r1, valid_loader_r2) = self.data_aug.get_data_loaders()

        optimizer = torch.optim.Adam(
            model.parameters(), self.config['optim']['init_lr'],
            weight_decay=self.config['optim']['weight_decay']
        )
        print('Optimizer:', optimizer)

        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['epochs'] - 9, eta_min=0, last_epoch=-1)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        # 设置断点的文件路径
        checkpoint_file = \
            '/home/xt/文档/jia/Infusion/Multi-fragments/model_checkpoints_folder/checkpoint.pkl'

        # 检查是否存在断点文件
        if os.path.exists(checkpoint_file):
            # 加载断点并继续训练
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                epoch_counter = checkpoint['epoch']
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                loss_list = checkpoint['loss_list']
                n_iter = checkpoint['n_iter']
                valid_n_iter = checkpoint['valid_n_iter']
                best_valid_loss = checkpoint['best_valid_loss']

        # 初始化GradScaler对象
        scaler = GradScaler()

        train_losses = []
        losses = []

        for epoch_counter in range(self.config['epochs']):
            print('The', epoch_counter, 'epoch is beginning!!!')
            starttime = datetime.datetime.now()
            for (g1, g2, mols, frag_mols), (g1_rec, g2_rec, mols, frag_mols) in zip(train_loader_b1, train_loader_r2):
                if g1 is not None and g2 is not None and g1_rec is not None and g2_rec is not None:

                    optimizer.zero_grad()

                    g1 = g1.to(self.device)
                    g2 = g2.to(self.device)

                    with autocast():
                        __, z1_global, z1_sub = model(g1)
                        __, z2_global, z2_sub = model(g2)

                        z1_global = F.normalize(z1_global, dim=1)
                        z2_global = F.normalize(z2_global, dim=1)

                        z1_sub = F.normalize(z1_sub, dim=1)
                        z2_sub = F.normalize(z2_sub, dim=1)

                        g1_rec = g1_rec.to(self.device)
                        g2_rec = g2_rec.to(self.device)

                        __, z1_rec_global, z1_rec_sub = model(g1_rec)
                        __, z2_rec_global, z2_rec_sub = model(g2_rec)

                        # z1_rec_global = F.normalize(z1_rec_global, dim=1)
                        z2_rec_global = F.normalize(z2_rec_global, dim=1)

                        z1_rec_sub = F.normalize(z1_rec_sub, dim=1)
                        z2_rec_sub = F.normalize(z2_rec_sub, dim=1)

                        nt_xent_global = WeightedNTXentLoss()
                        nt_xent = NTXentLoss()

                        loss_br_global = nt_xent_global.forward(z1_global, z2_rec_global, mols)
                        loss_bb_global = nt_xent_global.forward(z1_global, z2_global, mols)
                        loss_rr_sub = nt_xent.forward(z1_rec_sub, z2_rec_sub)
                        loss_bb_sub = nt_xent.forward(z1_sub, z2_sub)

                        loss_global = 0.5 * loss_br_global + 0.5 * loss_bb_global
                        loss_sub = 0.5 * loss_bb_sub + 0.5 * loss_rr_sub

                        reg_loss = elastic_net_reg(model, l1_lambda=0.0001, l2_lambda=0.0001)

                        loss = (loss_global + reg_loss) + 0.5 * (loss_sub + reg_loss)
                        train_losses.append(loss)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if n_iter % self.config['log_every_n_steps'] == 0:
                        self.writer.add_scalar('loss_global', loss_global.item(), global_step=n_iter)
                        self.writer.add_scalar('loss_sub', loss_sub.item(), global_step=n_iter)
                        self.writer.add_scalar('loss', loss.item(), global_step=n_iter)
                        self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                        # print('epoch', epoch_counter, ': ' 'train_loss(total):', loss.item())

                    n_iter = n_iter + 1

            train_loss_fin = train_losses[-1]

            val_reg_loss = elastic_net_reg(model, l1_lambda=0.0001, l2_lambda=0.00001)

            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss_global, valid_loss_sub = self._validate(model, valid_loader_b1, valid_loader_r2)
                valid_loss = (valid_loss_global + val_reg_loss) + 0.5 * (valid_loss_sub + val_reg_loss)

                print(epoch_counter, 'valid_loss:', valid_loss)
                losses.append(valid_loss)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(),
                               '../.pth')

                self.writer.add_scalar('valid_loss_global', valid_loss_global, global_step=valid_n_iter)
                self.writer.add_scalar('valid_loss_sub', valid_loss_sub, global_step=valid_n_iter)
                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter = valid_n_iter + 1

            if (epoch_counter + 1) % 5 == 0:
                torch.save(model.state_dict(),
                           os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            if epoch_counter >= self.config['warmup'] - 1:
                scheduler.step()

            checkpoint = {
                'epoch': epoch_counter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_list': loss_list,
                'n_iter': n_iter,
                'valid_n_iter': valid_n_iter,
                'best_valid_loss': best_valid_loss
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)

            endtime = datetime.datetime.now()

            print('Epoch time:', (endtime - starttime).seconds / 60, 'minutes')
            train_losses.clear()

    def _validate(self, model, valid_loader_b1, valid_loader_r2):
        with torch.no_grad():
            model.eval()

            valid_loss_global, valid_loss_sub = 0.0, 0.0
            counter = 0
            for (g1, g2, mols, frag_mols), (g1_rec, g2_rec, mols, frag_mols) in zip(valid_loader_b1, valid_loader_r2):
                if g1 is not None and g2 is not None:
                    g1 = g1.to(self.device)
                    g2 = g2.to(self.device)

                    __, z1_global, z1_sub = model(g1)
                    __, z2_global, z2_sub = model(g2)

                    z1_global = F.normalize(z1_global, dim=1)
                    z2_global = F.normalize(z2_global, dim=1)

                    z1_sub = F.normalize(z1_sub, dim=1)
                    z2_sub = F.normalize(z2_sub, dim=1)

                if g1_rec is not None and g2_rec is not None:
                    g1_rec = g1_rec.to(self.device)
                    g2_rec = g2_rec.to(self.device)

                    __, z1_rec_global, z1_rec_sub = model(g1_rec)
                    __, z2_rec_global, z2_rec_sub = model(g2_rec)

                    # z1_rec_global = F.normalize(z1_rec_global, dim=1)
                    z2_rec_global = F.normalize(z2_rec_global, dim=1)

                    z1_rec_sub = F.normalize(z1_rec_sub, dim=1)
                    z2_rec_sub = F.normalize(z2_rec_sub, dim=1)

                    nt_xent_global = WeightedNTXentLoss()
                    nt_xent = NTXentLoss()

                    valid_loss_br_global = nt_xent_global.forward(z1_global, z2_rec_global, mols)
                    valid_loss_bb_global = nt_xent_global.forward(z1_global, z2_global, mols)
                    valid_loss_rr_sub = nt_xent.forward(z1_rec_sub, z2_rec_sub)
                    valid_loss_bb_sub = nt_xent.forward(z1_sub, z2_sub)
                    counter = counter + 1

                valid_loss_global = 0.5 * valid_loss_br_global + 0.5 * valid_loss_bb_global
                valid_loss_sub = 0.5 * valid_loss_bb_sub + 0.5 * valid_loss_rr_sub
                valid_loss_global /= counter
                valid_loss_sub /= counter

        model.train()

        return valid_loss_global, valid_loss_sub


def main():
    config = yaml.load(open("../config.yaml", "r"),
                       Loader=yaml.FullLoader)
    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    molclr = fragCLR(dataset, config)
    molclr.train()


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    main()

    end_time = datetime.datetime.now()

    print('Total running time:', (end_time - start_time).seconds / 60, 'minutes')
