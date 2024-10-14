import time
import datetime
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from config import get_config
from model.data_aug2 import build_loader
from model.data_aug_test import MolTestDatasetWrapper
from model.frag_test import get_model, get_contrast_config, step
from model.loss import build_loss
from utils import create_logger, seed_set, save_best_checkpoint
from utils import NoamLR, build_scheduler, build_optimizer, get_metric_func
from utils import load_checkpoint, load_best_result
from model.Hier_Net import build_model, get_device
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)
data_b_list, data_m_list = [], []


def parse_args():
    parser = argparse.ArgumentParser(description="codes for HiGNN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="../configs/bbbp.yaml",
        type=str,
    )

    parser.add_argument(
        "--opts",
        help=" Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--batch-size', type=int, help="batch size for training")
    parser.add_argument('--lr_scheduler', type=str, help='learning rate scheduler')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    # MacFrag config
    parser.add_argument('-input_file', '-i', required=True,
                        help='.smi or .sdf file of molecules to be fragmented')

    parser.add_argument('-output_path', '-o', required=True,
                        help='path of the output fragments file')

    parser.add_argument('-maxBlocks', required=True,
                        help='the maximum number of building blocks that the fragments contain')

    parser.add_argument('-maxSR', required=True,
                        help='only cyclic bonds in smallest SSSR ring of size larger than this value will be cleaved')

    parser.add_argument('-asMols', required=True,
                        help='True of False; if True, '
                             'MacFrag will reture fragments as molecules and the fragments.'
                             'sdf file will be output; if False, MacFrag will reture fragments.'
                             'smi file with fragments representd as SMILES strings')

    parser.add_argument('-minFragAtoms', required=True,
                        help='the minimum number of atoms that the fragments contain')

    args = parser.parse_args()
    cfg = get_config(args)

    return args, cfg


out_b_list, out_m_list = [], []


def get_contrast_representations(data):
    contrast_model = get_model(contrast_config)
    contrast_representations = contrast_model(data)
    return contrast_representations


def train_one_epoch(cfg, model_contrast, model, criterion, train_loader_contrast, trainloader, optimizer, lr_scheduler, device, logger):
    model.train()
    loss_contrast = []
    losses = []
    y_pred_b_list = {}
    y_label_b_list = {}

    features = []  # feature vectors
    labels = []

    for data, data_b in zip(train_loader_contrast, trainloader):
        data = data.to(device)
        loss = step(model_contrast, data)
        loss_contrast.append(loss)
        output_b = model.forward(data_b)

        _, mol_vec, _ = output_b
        features.append(mol_vec.cpu())
        labels.extend(data.y.cpu().numpy())

        if isinstance(output_b, tuple):
            output_b, vec1_b, vec2_b = output_b
            # output_m, vec1_m, vec2_m = output_m
        else:
            output_b, vec1_b, vec2_b = output_b, None, None
            # output_m, vec1_m, vec2_m = output_m, None, None

        loss_b = 0

        for i in range(1):
            if cfg.DATA.TASK_TYPE == 'classification':
                if output_b.dim() == 1:
                    continue
                else:
                    y_pred_b = output_b[:, i * 2:(i + 1) * 2].to(device)
                    y_label_b = data_b.y[i].squeeze().to(device)
                    validId = np.where((y_label_b.cpu().numpy() == 0) | (y_label_b.cpu().numpy() == 1))[0]
                    if y_label_b.dim() == 0:
                        y_label_b = y_label_b.unsqueeze(0)
                    y_pred_b = y_pred_b[torch.tensor(validId).to(device)]
                    y_label_b = y_label_b[torch.tensor(validId).to(device)]

                if isinstance(criterion, list):
                    for c in criterion:
                        loss_b += c(y_pred_b, y_label_b, vec1_b, vec2_b).to(get_device())
                else:
                    loss_b += criterion(y_pred_b, y_label_b, vec1_b, vec2_b).to(get_device())

                loss_infusion = 0.5 * loss_b + 0.5 * loss
                losses.append(loss_infusion)

                try:
                    y_label_b_list[i].extend(y_label_b.cpu().numpy())
                    y_pred_b_list[i].extend(y_pred_b)
                except:
                    y_label_b_list[i] = []
                    y_pred_b_list[i] = []
                    y_label_b_list[i].extend(y_label_b.cpu().numpy())
                    y_pred_b_list[i].extend(y_pred_b)

            elif cfg.DATA.TASK_TYPE == 'regression':
                y_pred_b = output_b[i][1].to(device)
                y_label_b = data_b.y[i].to(device)

                if y_label_b.dim() == 0:
                    y_label_b = y_label_b.unsqueeze(0)

                if isinstance(criterion, list):
                    for c in criterion:
                        loss_b += c(y_pred_b, y_label_b).to(device)
                else:
                    loss_b += criterion(y_pred_b, y_label_b).to(device)

                loss_infusion = 0.5 * loss_b + 0.5 * loss
                losses.append(loss_infusion)

                try:
                    y_label_b_list[i].extend(y_label_b.cpu().numpy().flatten())
                    y_pred_b_list[i].extend(y_pred_b.cpu().numpy().flatten())
                except:
                    y_label_b_list[i] = []
                    y_pred_b_list[i] = []
                    y_label_b_list[i].extend(y_label_b.cpu().numpy().flatten())
                    y_pred_b_list[i].extend(y_pred_b.cpu().detach().numpy().flatten())

        del output_b, vec1_b, vec2_b
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        if not isinstance(loss_infusion, int):
            loss_infusion.backward()
            optimizer.step()
            losses.append(loss_infusion)

        if isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step()

    # Compute metric
    results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i in range(1):
        y_label_b_list[i] = np.array(y_label_b_list[i])
        y_pred_b_list[i] = [pred[1].item() for pred in y_pred_b_list[i]]
        results.append(metric_func(y_label_b_list[i], y_pred_b_list[i]))

    avg_results = np.nanmean(results)
    trn_loss = np.array([loss.cpu().detach().numpy() for loss in losses]).mean()

    # # regression tasks
    # for i in range(1):
    #     if len(y_label_b_list[i]) == 0:
    #         continue
    #
    #     for p in y_pred_b_list[i]:
    #         results.append(metric_func([p], y_label_b_list[i]))
    #
    # avg_results = torch.mean(torch.tensor(results)).item() if isinstance(
    #     results[0], torch.Tensor) else np.mean(results)
    #
    # trn_loss = np.array([loss.cpu().detach().numpy() for loss in losses]).mean()

    return trn_loss, avg_results


def validate(cfg, model_contrast, model, criterion, data_loader_contrast, dataloader, epoch, device, logger, eval_mode=False):
    model.eval()
    loss_contrast = []
    losses = []
    y_pred_list = {}
    y_label_list = {}

    for data, data_b in zip(data_loader_contrast, dataloader):
        data = data.to(device)
        loss = step(model_contrast, data)
        loss_contrast.append(loss)
        data_b = data_b.to(device)
        output_b = model.forward(data_b)

        if isinstance(output_b, tuple):
            output_b, vec1_b, vec2_b = output_b
            # output_m, vec1_m, vec2_m = output_m
        else:
            output_b, vec1_b, vec2_b = output_b, None, None
            # output_m, vec1_m, vec2_m = output_m, None, None

        loss_b = 0

        for i in range(1):
            if cfg.DATA.TASK_TYPE == 'classification':
                if output_b.dim() == 1:
                    continue
                else:
                    y_pred_b = output_b[:, i * 2:(i + 1) * 2].to(device)
                    y_label_b = data_b.y[i].squeeze().to(device)
                    validId = np.where((y_label_b.cpu().numpy() == 0) | (y_label_b.cpu().numpy() == 1))[0]
                    if y_label_b.dim() == 0:
                        y_label_b = y_label_b.unsqueeze(0)
                    y_pred_b = y_pred_b[torch.tensor(validId).to(device)]
                    y_label_b = y_label_b[torch.tensor(validId).to(device)]

                if isinstance(criterion, list):
                    for c in criterion:
                        loss_b += c(y_pred_b, y_label_b, vec1_b, vec2_b).to(get_device())
                else:
                    loss_b += criterion(y_pred_b, y_label_b, vec1_b, vec2_b).to(get_device())

                loss_infusion = 0.5 * loss_b + 0.5 * loss
                losses.append(loss_infusion)

                try:
                    y_label_list[i].extend(y_label_b.cpu().numpy())
                    y_pred_list[i].extend(y_pred_b)
                except:
                    y_label_list[i] = []
                    y_pred_list[i] = []
                    y_label_list[i].extend(y_label_b.cpu().numpy())
                    y_pred_list[i].extend(y_pred_b)

            elif cfg.DATA.TASK_TYPE == 'regression':
                y_pred_b = output_b[i][1].to(device)
                y_label_b = data_b.y[i].to(device)

                if y_label_b.dim() == 0:
                    y_label_b = y_label_b.unsqueeze(0)

                if isinstance(criterion, list):
                    for c in criterion:
                        loss_b += c(y_pred_b, y_label_b).to(device)
                else:
                    loss_b += criterion(y_pred_b, y_label_b).to(device)

                loss_infusion = 0.5 * loss_b + 0.5 * loss
                losses.append(loss_infusion)

                try:
                    y_label_list[i].extend(y_label_b.cpu().numpy().flatten())
                    y_pred_list[i].extend(y_pred_b.cpu().numpy().flatten())
                except:
                    y_label_list[i] = []
                    y_pred_list[i] = []
                    y_label_list[i].extend(y_label_b.cpu().numpy().flatten())
                    y_pred_list[i].extend(y_pred_b.cpu().detach().numpy().flatten())

        del output_b, vec1_b, vec2_b
        torch.cuda.empty_cache()

    # Compute metric
    val_results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i in range(1):
        if len(y_label_list) > 0:
            y_label_list[i] = np.array(y_label_list[i])
            y_pred_list[i] = [pred[1].item() for pred in y_pred_list[i]]
            val_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_val_results = np.nanmean(val_results)
    val_loss = np.array([loss.cpu().detach().numpy() for loss in losses]).mean()

    # # regression
    # for i in range(len(y_label_list)):
    #     if len(y_label_list[i]) == 0:
    #         continue
    #
    #     # y_pred_list[i] = [find_nearest_value(y_pred_list[i], y_label_list[i][0])]
    #     val_results.append(metric_func(y_pred_list[i], y_label_list[i]))
    #
    # avg_val_results = np.nanmean(val_results)
    # val_loss = np.array([loss.cpu().detach().numpy() for loss in losses]).mean()

    # if eval_mode:
    #     logger.info(f'Seed {cfg.SEED} Dataset {cfg.DATA.DATASET} ==> '
    #                 f'The best epoch:{epoch} test_loss:{val_loss:.3f} test_scores:{avg_val_results:.3f}')
    #     return val_results

    return val_loss, avg_val_results


def test(cfg, model_contrast, model, criterion, data_loader_contrast, dataloader, epoch, device, logger, eval_mode=False):
    model.eval()
    loss_contrast = []
    losses = []
    y_pred_list = {}
    y_label_list = {}

    features = []  # feature vectors
    labels = []

    for data, data_b in zip(data_loader_contrast, dataloader):
        data = data.to(device)
        loss = step(model_contrast, data)
        loss_contrast.append(loss)
        data_b = data_b.to(device)
        output_b = model.forward(data_b)

        mol_vec = output_b
        features.append(mol_vec[0].cpu())
        labels.extend(data.y.cpu().numpy())

        if isinstance(output_b, tuple):
            output_b, vec1_b, vec2_b = output_b
            # output_m, vec1_m, vec2_m = output_m
        else:
            output_b, vec1_b, vec2_b = output_b, None, None
            # output_m, vec1_m, vec2_m = output_m, None, None

        loss_b = 0

        for i in range(1):
            if cfg.DATA.TASK_TYPE == 'classification':
                if output_b.dim() == 1:
                    continue
                else:
                    y_pred_b = output_b[:, i * 2:(i + 1) * 2].to(device)
                    y_label_b = data_b.y[i].squeeze().to(device)
                    validId = np.where((y_label_b.cpu().numpy() == 0) | (y_label_b.cpu().numpy() == 1))[0]
                    if y_label_b.dim() == 0:
                        y_label_b = y_label_b.unsqueeze(0)
                    y_pred_b = y_pred_b[torch.tensor(validId).to(device)]
                    y_label_b = y_label_b[torch.tensor(validId).to(device)]

                if isinstance(criterion, list):
                    for c in criterion:
                        loss_b += c(y_pred_b, y_label_b, vec1_b, vec2_b).to(get_device())
                else:
                    loss_b += criterion(y_pred_b, y_label_b, vec1_b, vec2_b).to(get_device())

                loss_infusion = 0.5 * loss_b + 0.5 * loss
                losses.append(loss_infusion)

                try:
                    y_label_list[i].extend(y_label_b.cpu().numpy())
                    y_pred_list[i].extend(y_pred_b)
                except:
                    y_label_list[i] = []
                    y_pred_list[i] = []
                    y_label_list[i].extend(y_label_b.cpu().numpy())
                    y_pred_list[i].extend(y_pred_b)

            elif cfg.DATA.TASK_TYPE == 'regression':
                y_pred_b = output_b[i][1].to(device)
                y_label_b = data_b.y[i].to(device)

                if y_label_b.dim() == 0:
                    y_label_b = y_label_b.unsqueeze(0)

                if isinstance(criterion, list):
                    for c in criterion:
                        loss_b += c(y_pred_b, y_label_b).to(device)
                else:
                    loss_b += criterion(y_pred_b, y_label_b).to(device)

                loss_infusion = 0.5 * loss_b + 0.5 * loss
                losses.append(loss_infusion)

                try:
                    y_label_list[i].extend(y_label_b.cpu().numpy().flatten())
                    y_pred_list[i].extend(y_pred_b.cpu().numpy().flatten())
                except:
                    y_label_list[i] = []
                    y_pred_list[i] = []
                    y_label_list[i].extend(y_label_b.cpu().numpy().flatten())
                    y_pred_list[i].extend(y_pred_b.cpu().detach().numpy().flatten())

    # Compute metric
    val_results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i in range(1):
        if len(y_label_list) > 0:
            y_label_list[i] = np.array(y_label_list[i])
            y_pred_list_visual = [pred[1] for pred in y_pred_list[i]]
            y_pred_list[i] = [pred[1].item() for pred in y_pred_list[i]]
            val_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_val_results = np.nanmean(val_results)
    val_loss = np.array([loss.cpu().detach().numpy() for loss in losses]).mean()

    # # regression
    # for i in range(len(y_label_list)):
    #     if len(y_label_list[i]) == 0:
    #         continue
    #
    #     # y_pred_list[i] = [find_nearest_value(y_pred_list[i], y_label_list[i][0])]
    #     val_results.append(metric_func(y_pred_list[i], y_label_list[i]))
    #
    # avg_val_results = np.nanmean(val_results)
    # val_loss = np.array([loss.cpu().detach().numpy() for loss in losses]).mean()

    # if eval_mode:
    #     logger.info(f'Seed {cfg.SEED} Dataset {cfg.DATA.DATASET} ==> '
    #                 f'The best epoch:{epoch} test_loss:{val_loss:.3f} test_scores:{avg_val_results:.3f}')
    #     return val_results

    return val_loss, avg_val_results, y_label_list[0], y_pred_list_visual


def compute_auc(y_true, y_pred):
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    if len(np.unique(y_true)) > 1:
        try:
            auc_score = metric_func(y_true, y_pred)
            return auc_score
        except ValueError as e:
            print(f"Error calculating AUC: {e}")
            return None
    else:
        print("Skipping AUC calculation due to single class present in y_true.")
        return None


def train(cfg, logger):
    seed_set(cfg.SEED)
    # step 1: dataloder 加载
    config, target_list = get_contrast_config()
    config['dataset']['target'] = target_list[0]
    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    train_loader_contrast, valid_loader_contrast, test_loader_contrast = \
        dataset.get_data_loaders()  # fragment-based contrastive learning
    train_loader, val_loader, test_loader, weights = build_loader(cfg, logger)  # multi-layer attention

    # step 2: 加载模型
    model_contrast = get_model(config)
    model = build_model(cfg)
    logger.info(model)
    device = get_device()
    model.to(device)

    # step 3: 加载优化器
    optimizer = build_optimizer(cfg, model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    # step 4: 学习率
    lr_scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    # step 5: loss函数
    if weights is not None:
        criterion = [build_loss(cfg, weights.to(device))]
    else:
        criterion = build_loss(cfg, weight=None)

    if cfg.TRAIN.TENSORBOARD.ENABLE:
        tensorboard_dir = os.path.join(cfg.OUTPUT_DIR, "tensorboard")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
    else:
        tensorboard_dir = None

    if tensorboard_dir is not None:
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        writer = None

    best_epoch, best_score = 0, 0 if cfg.DATA.TASK_TYPE == 'classification' else float('inf')
    if cfg.TRAIN.RESUME:
        best_epoch, best_score = load_checkpoint(cfg, model, optimizer, lr_scheduler, logger)
        validate(cfg, model, criterion, val_loader, best_epoch, device, logger)

        if cfg.EVAL_MODE:
            return

    logger.info("Start training")
    early_stop_cnt = 0
    start_time = time.time()

    y_label_list_all, y_pred_list_all = [], []

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCHS):

        # 1: 训练单个epoch
        # print(logger)
        trn_loss, trn_score = train_one_epoch(cfg, model_contrast, model, criterion, train_loader_contrast, train_loader, optimizer,
                                              lr_scheduler,
                                              device, logger)

        val_loss, val_score = validate(cfg, model_contrast, model, criterion, valid_loader_contrast, val_loader
                                       , epoch, device, logger)

        test_loss, test_score, y_label_list, y_pred_list = test(cfg, model_contrast, model, criterion, test_loader_contrast, test_loader
                                         , epoch, device, logger)
        y_label_list_all.extend(y_label_list)
        y_pred_list_all.extend(y_pred_list)

        # 2: 更新学习率
        if not isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step(val_loss)

        # 3: 记录单个epoch结果
        if epoch % cfg.SHOW_FREQ == 0 or epoch == cfg.TRAIN.MAX_EPOCHS - 1:
            lr_cur = lr_scheduler.optimizer.param_groups[0]['lr']
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} trn_loss:{trn_loss:.3f} '
                        f'trn_{cfg.DATA.METRIC}:{trn_score:.3f} lr:{lr_cur:.5f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} val_loss:{val_loss:.3f} '
                        f'val_{cfg.DATA.METRIC}:{val_score:.3f} lr:{lr_cur:.5f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} test_loss:{test_loss:.3f} '
                        f'test_{cfg.DATA.METRIC}:{test_score:.3f} lr:{lr_cur:.5f}')

        loss_dict, acc_dict = {"train_loss": trn_loss}, {f"train_{cfg.DATA.METRIC}": trn_score}
        loss_dict["valid_loss"], acc_dict[f"valid_{cfg.DATA.METRIC}"] = val_loss, val_score

        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars(f"scalar/{cfg.DATA.METRIC}", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)

        if cfg.DATA.TASK_TYPE == 'classification' and val_score > best_score or \
                cfg.DATA.TASK_TYPE == 'regression' and val_score < best_score:
            best_score, best_epoch = val_score, epoch
            save_best_checkpoint(cfg, epoch, model, best_score, best_epoch, optimizer, lr_scheduler, logger)
            early_stop_cnt = 0
        else:
            early_stop_cnt = 1

        if early_stop_cnt > cfg.TRAIN.EARLY_STOP > 0:
            logger.info('Early stop hitted!')
            break

    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')

    model, best_epoch = load_best_result(cfg, model, logger)
    score = validate(cfg, model_contrast, model, criterion, test_loader_contrast, test_loader, best_epoch, device, logger=logger, eval_mode=True)

    return score


if __name__ == "__main__":
    contrast_config, target_list = get_contrast_config()
    _, cfg = parse_args()

    logger = create_logger(cfg)

    logger.info(cfg.dump())

    # print device mode
    logger.info('GPU mode...')
    print('The logger is below:', logger)

    # training
    score = train(cfg, logger)
