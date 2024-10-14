import os
import yaml
import torch
import pandas as pd
from datetime import datetime
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score
from model.GINet_finetune import gnnet
from model.data_aug_test import MolTestDatasetWrapper


def save_config_file(log_dir, config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        with open(os.path.join(log_dir, 'config_test.yaml'), 'w') as config_file:
            yaml.dump(config, config_file)


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


loss_list = []
roc_auc_values = []


def get_model(configs):
    model_config = yaml.load(
        open('../config_test.yaml', "r"),
        Loader=yaml.FullLoader)
    model_config = model_config['model']
    model_config['dropout'] = configs['model']['dropout']
    model_config['pool'] = configs['model']['pool']
    model = gnnet(configs['dataset']['task'], **model_config).to('cuda:0')
    # model = _load_pre_trained_weights(model)
    return model


def _load_pre_trained_weights(model):
    try:
        checkpoints_folder = os.path.join(
            '../model_checkpoints_folder/')
        ckp_path = os.path.join(checkpoints_folder, 'model.pth')
        state_dict = torch.load(ckp_path, map_location='cuda:0')

        own_state = state_dict
        for name, param in state_dict.items():
            if name not in own_state:
                print('NOT LOADED:', name)
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            own_state[name].copy_(param)
        print("Loaded pre-trained model {} with success.".format(ckp_path))
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")

    return model


class FineTune(object):
    def __init__(self, dataset2, config):
        self.config = config
        self.device = self._get_device()
        self.dataset2 = dataset2

        dir_name = config['fine_tune_from'].split('/')[0] + '-' + \
                   config['fine_tune_from'].split('/')[-1] + '-' + config['task_name']

        self.log_dir = os.path.join('experiments', dir_name)

        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8']:
                self.criterion = nn.SmoothL1Loss()
            else:
                self.criterion = nn.MSELoss()

        save_config_file(self.log_dir, self.config)

    def _get_device(self):
        device = 'cuda:0'
        return device

    def _step(self, model, data):
        pred = model(data)
        if self.config['dataset']['task'] == 'classification':
            print('pred:', pred.shape, 'data.y:', data.y.squeeze(dim=1).shape, data.y.squeeze(dim=1))
            loss = self.criterion(pred, data.y.view(-1))
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred[:, 1], data.y.view(-1))

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset2.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7']:
            labels = []
            for d in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        n_batches = len(train_loader)
        if n_batches < self.config['log_every_n_steps']:
            self.config['log_every_n_steps'] = n_batches

        model = get_model(self.config)

        layer_list = []
        for name, param in model.named_parameters():
            if 'output_layers' in name:
                print(name)
                layer_list.append(name)

        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        if self.config['optim']['type'] == 'SGD':
            init_lr = self.config['optim']['base_lr'] * self.config['batch_size'] / 256
            optimizer = torch.optim.SGD(
                [{'params': params, 'lr': init_lr},
                 {'params': base_params, 'lr': init_lr * self.config['optim']['base_ratio']}
                 ],
                momentum=self.config['optim']['momentum'],
                weight_decay=self.config['optim']['weight_decay']
            )
        elif self.config['optim']['type'] == 'Adam':
            optimizer = torch.optim.Adam(
                [{'params': params, 'lr': self.config['optim']['lr']},
                 {'params': base_params, 'lr': self.config['optim']['lr'] * self.config['optim']['base_ratio']}
                 ],
                weight_decay=self.config['optim']['weight_decay']
            )
        else:
            raise ValueError('Not defined optimizer type!')

        n_iter = 0
        valid_n_iter = 0
        best_valid_rmse = np.inf
        best_valid_mae = np.inf
        best_valid_roc_auc = 0

        for epoch_counter in range(self.config['epochs']):
            print('It is the', epoch_counter, 'epoch:')
            for bn, data in enumerate(train_loader):
                data = data.to(self._get_device())
                loss = self._step(model, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n_iter += 1

            # validate
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification':
                    valid_loss, valid_roc_auc = self._validate(model, valid_loader)
                    if valid_roc_auc > best_valid_roc_auc:
                        best_valid_roc_auc = valid_roc_auc
                        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model_br.pth'))
                elif self.config['dataset']['task'] == 'regression':
                    valid_loss, valid_rmse, valid_mae = self._validate(model, valid_loader)
                    if self.config["task_name"] in ['qm7', 'qm8'] and valid_mae < best_valid_mae:
                        best_valid_mae = valid_mae
                        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model_br.pth'))
                    elif valid_rmse < best_valid_rmse:
                        best_valid_rmse = valid_rmse
                        torch.save(model.state_dict(), os.path.join(self.log_dir, 'model_br.pth'))

                valid_n_iter += 1

        return self._test(model, test_loader)

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)
                pred = model(data)
                loss = self._step(model, data)
                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    if self.config["task_name"] in ['BBBP', 'BACE']:
                        pred = torch.sigmoid(pred[:, 1]).squeeze()
                    else:
                        pred = F.softmax(pred, dim=-1)
                elif self.config['dataset']['task'] == 'regression':
                    pred = F.softmax(pred, dim=-1)

                if pred.dim() == 0:
                    pred = pred.view(1)

                predictions.extend(pred.cpu().detach().numpy())
                labels.extend(data.y.cpu().flatten().numpy())
                valid_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            rmse = mean_squared_error(labels, predictions[:, 0], squared=False)
            mae = mean_absolute_error(labels, predictions[:, 0])
            print('Validation loss:', valid_loss, 'RMSE:', rmse, 'MAE:', mae)
            return valid_loss, rmse, mae

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)

            if self.config["task_name"] in ['BBBP', 'BACE']:
                roc_auc = roc_auc_score(labels, predictions)
            else:
                roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Validation loss:', valid_loss, 'roc_auc:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        predictions = []
        labels = []

        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            num_data = 0

            for bn, data in enumerate(test_loader):
                data = data.to(self.device)
                pred = model(data)
                loss = self._step(model, data)
                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    if self.config["task_name"] in ['BBBP', 'BACE']:
                        pred = torch.sigmoid(pred[:, 1]).squeeze()
                    else:
                        pred = F.softmax(pred[:, 1], dim=-1)
                elif self.config['dataset']['task'] == 'regression':
                    pass

                if pred.dim() != 0 and data.y.dim() != 0:
                    predictions.extend(pred.cpu())
                    labels.extend(data.y.cpu().flatten().numpy())

        test_loss /= num_data
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array([p[1].item() for p in predictions])
            labels = np.array(labels)
            rmse = mean_squared_error(labels, predictions, squared=False)
            mae = mean_absolute_error(labels, predictions)
            print('Test loss:', test_loss, 'RMSE:', rmse, 'MAE:', mae)
            return test_loss, rmse, mae

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions)
            print('Test loss:', test_loss, 'ROC AUC:', roc_auc)
            return test_loss, roc_auc


def step(model, data):
    config = yaml.load(
        open("../config_test.yaml", "r"),
        Loader=yaml.FullLoader)
    pred = model(data)
    if config['dataset']['task'] == 'classification':
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, data.y.view(-1))
    elif config['dataset']['task'] == 'regression':
        if config["task_name"] in ['qm7', 'qm8']:
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        loss = criterion(pred, data.y)
    return loss


class FineTune_loss(object):
    def __init__(self, dataset2, config):
        self.config = config
        self.device = self._get_device()
        self.dataset2 = dataset2

        dir_name = config['fine_tune_from'].split('/')[0] + '-' + \
                   config['fine_tune_from'].split('/')[-1] + '-' + config['task_name']
        self.log_dir = os.path.join('experiments', dir_name)

        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8']:
                # self.criterion = nn.L1Loss()
                self.criterion = nn.SmoothL1Loss()
            else:
                self.criterion = nn.MSELoss()

        save_config_file(self.log_dir, self.config)

    def _get_device(self):
        device = 'cuda:0'
        return device

    def train_loss(self):
        train_loader, valid_loader, test_loader = self.dataset2.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7']:
            labels = []
            for d in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        n_batches = len(train_loader)
        if n_batches < self.config['log_every_n_steps']:
            self.config['log_every_n_steps'] = n_batches

        model = get_model(self.config)

        layer_list = []
        for name, param in model.named_parameters():
            if 'output_layers' in name:
                print(name)
                layer_list.append(name)

        for bn, data in enumerate(train_loader):
            data = data.to(self._get_device())
            loss = self._step(model, data)

        return loss

    def validate_loss(self, model, valid_loader):
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)
                loss = self._step(model, data)
                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)
            valid_loss /= num_data

        return valid_loss

    def test_loss(self, model, test_loader):

        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            num_data = 0

            for bn, data in enumerate(test_loader):
                data = data.to(self.device)
                loss = self._step(model, data)
                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

        test_loss /= num_data
        return test_loss


def run(config):
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune(dataset, config)
    return fine_tune.train()


def get_contrast_config():
    config = yaml.load(
        open("/home/xt/文档/jia/Infusion/Multi-fragments/pretrained/checkpoints/config_test.yaml", "r"),
        Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/CC50_train.csv'
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/Tox21/raw/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/ClinTox/raw/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/HIV/raw/hiv.csv'
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/bace/raw/bace.csv'
        target_list = ["Class"]

    elif config['task_name'] == 'ToxCast':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/ToxCast/raw/toxcast.csv'
        target_list = ['ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive',
                       'APR_HepG2_CellCycleArrest_24h_dn', 'APR_HepG2_CellCycleArrest_72h_dn',
                       'APR_HepG2_CellLoss_24h_dn', 'APR_HepG2_CellLoss_72h_dn']

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = '/h ome/xt/文档/jia/Infusion/Multi-fragments/data/SIDER/raw/sider.csv'
        target_list = [
            "Blood and lymphatic system disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders",
            "Investigations",
            "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances",
            "Immune system disorders", "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions",
            "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders",
            "",
            "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders",
            "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders",
            "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders",
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]

    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/muv/raw/muv.csv'
        target_list = [
            "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-692", "MUV-712", "MUV-713",
            "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/FreeSolv/raw/freesolv.csv'
        target_list = ["freesolv"]

    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/ESOL/raw/esol.csv'
        target_list = ["ESOL predicted log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/Lipo/raw/lipo.csv'
        target_list = ["lipo"]

    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0",
            "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"
        ]

    elif config["task_name"] == 'qm9':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = '/home/xt/文档/jia/Infusion/Multi-fragments/data/qm9.csv'
        target_list = [
            "A", "B", "C",
            "mu", "alpha", "homo", "lumo", "gap", "r2",
            "zpve", "u0", "u298", "h298", "g298", "cv",
            "u0_atom", "u298_atom", "h298_atom", "g298_atom"
        ]

    else:
        raise ValueError('Unspecified dataset!')

    print(config)
    return config, target_list


if __name__ == '__main__':
    config, target_list = get_contrast_config()

    os.makedirs('experiments', exist_ok=True)
    dir_name = config['fine_tune_from'].split('/')[0] + '-' + \
               config['fine_tune_from'].split('/')[-1] + '-' + config['task_name']
    save_dir = os.path.join('experiments', dir_name)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    if config['dataset']['task'] == 'classification':
        save_list = []
        for target in target_list:
            config['dataset']['target'] = target
            roc_list = [target]
            test_loss, roc_auc = run(config)
            roc_list.append(roc_auc)
            save_list.append(roc_list)

        print('The task', config["task_name"], 'roc_auc:', save_list)

        df = pd.DataFrame(save_list)
        fn = '{}_{}_ROC.csv'.format(config["task_name"], current_time)
        df.to_csv(os.path.join(save_dir, fn), index=False, header=['label', 'ROC-AUC'])

    elif config['dataset']['task'] == 'regression':
        save_rmse_list, save_mae_list = [], []
        for target in target_list:
            config['dataset']['target'] = target
            rmse_list, mae_list = [target], [target]
            test_loss, rmse, mae = run(config)
            rmse_list.append(rmse)
            mae_list.append(mae)
            save_rmse_list.append(rmse_list)
            save_mae_list.append(mae_list)

        print('The task', config["task_name"], 'roc_auc:', min(save_rmse_list))

        df = pd.DataFrame(save_rmse_list)
        fn = '{}_{}_RMSE.csv'.format(config["task_name"], current_time)
        df.to_csv(os.path.join(save_dir, fn), index=False, header=['label', 'RMSE'])

        df = pd.DataFrame(save_mae_list)
        fn = '{}_{}_MAE.csv'.format(config["task_name"], current_time)
        df.to_csv(os.path.join(save_dir, fn), index=False, header=['label', 'MAE'])
