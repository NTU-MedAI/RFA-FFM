import os
import yaml
import numpy as np
from copy import deepcopy
from hyperopt import fmin, hp, tpe
import torch
from Hier_Net import build_model
from Hier_train import train, parse_args
from utils import create_logger, get_task_names


def cross_validate(cfg, logger):

    init_seed = cfg.SEED
    out_dir = cfg.OUTPUT_DIR
    task_names = get_task_names(os.path.join(cfg.DATA.DATA_PATH, 'raw/{}.csv'.format(cfg.DATA.DATASET)))

    all_scores = []
    for fold_num in range(cfg.NUM_FOLDS):
        cfg.defrost()
        cfg.SEED = init_seed + fold_num
        cfg.OUTPUT_DIR = os.path.join(out_dir, f'fold_{fold_num}')
        cfg.freeze()
        logger.info(f'Fold {fold_num}')
        model_scores = train(cfg, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    cfg.defrost()
    cfg.OUTPUT_DIR = out_dir
    cfg.freeze()
    logger.info(f'{cfg.NUM_FOLDS}-fold cross validation')

    for fold_num, scores in enumerate(all_scores):
        logger.info(f'Seed {init_seed + fold_num} ==> test {cfg.DATA.METRIC} = {np.nanmean(scores):.3f}')
        if cfg.SHOW_EACH_SCORES:
            for task_name, score in zip(task_names, scores):
                logger.info(f'Seed {init_seed + fold_num} ==> test {task_name} {cfg.DATA.METRIC} = {score:.3f}')

    avg_scores = np.nanmean(all_scores, axis=1)
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    logger.info(f'Overall test {cfg.DATA.METRIC} = {mean_score:.3f} ± {std_score:.3f}')

    if cfg.SHOW_EACH_SCORES:
        for task_num, task_name in enumerate(task_names):
            logger.info(f'Overall test {task_name} {cfg.DATA.METRIC} = '
                        f'{np.nanmean(all_scores[:, task_num]):.3f} ± {np.nanstd(all_scores[:, task_num]):.3f}')

    return mean_score, std_score


SPACE = {
    'MODEL.HID1': hp.choice('dim', [64, 128, 256]),
    'MODEL.SLICES': hp.choice('slices', [1, 2, 4]),
    'MODEL.DROPOUT': hp.quniform('dropout', low=0.0, high=0.5, q=0.1),
    'MODEL.DEPTH': hp.choice('depth', [2, 3, 4]),
    'TRAIN.OPTIMIZER.BASE_LR': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
    'TRAIN.OPTIMIZER.WEIGHT_DECAY': hp.choice('l2', [1e-4, 1e-5, 1e-6]),
}
INT_KEYS = ['MODEL.HID1', 'MODEL.DEPTH', 'MODEL.SLICES']


def hyperopt(cfg, logger):

    yaml_name = "best_{}_{}.yaml".format(cfg.DATA.DATASET, cfg.TAG)
    cfg_save_path = os.path.join(cfg.OUTPUT_DIR, yaml_name)

    results = []

    def objective(hyperparams):
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        hyper_cfg = deepcopy(cfg)
        if hyper_cfg.OUTPUT_DIR is not None:
            folder_name = f'round_{hyper_cfg.HYPER_COUNT}'
            hyper_cfg.defrost()
            hyper_cfg.OUTPUT_DIR = os.path.join(hyper_cfg.OUTPUT_DIR, folder_name)
            hyper_cfg.freeze()
        hyper_cfg.defrost()
        opts = list()
        for key, value in hyperparams.items():
            opts.append(key)
            opts.append(value)
        hyper_cfg.merge_from_list(opts)
        hyper_cfg.freeze()

        cfg.defrost()
        cfg.HYPER_COUNT += 1
        cfg.freeze()
        logger.info(f'round_{hyper_cfg.HYPER_COUNT - 1}')
        logger.info(hyperparams)

        mean_score, std_score = cross_validate(hyper_cfg, logger)

        temp_model = build_model(hyper_cfg)
        num_params = sum(param.numel() for param in temp_model.parameters() if param.requires_grad)
        logger.info(f'num params: {num_params:,}')
        logger.info(f'{mean_score} ± {std_score} {hyper_cfg.DATA.METRIC}')

        results.append({
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams,
            'num_params': num_params
        })

        if np.isnan(mean_score):
            if hyper_cfg.DATA.TASK_TYPE == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')

        return (-1 if hyper_cfg.DATA.TASK_TYPE == 'classification' else 1) * mean_score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=cfg.NUM_ITERS, verbose=False)

    results = [result for result in results if not np.isnan(result['mean_score'])]
    best_result = \
        min(results, key=lambda result: (-1 if cfg.DATA.TASK_TYPE == 'classification' else 1) * result['mean_score'])
    logger.info('best result')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(f'{best_result["mean_score"]} ± {best_result["std_score"]} {cfg.DATA.METRIC}')

    with open(cfg_save_path, 'w') as f:
        yaml.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)





if __name__ == '__main__':
    _, cfg = parse_args()

    logger = create_logger(cfg)

    if torch.cuda.is_available():
        logger.info('GPU mode...')
    else:
        logger.info('CPU mode...')

    if cfg.HYPER:

        if cfg.MODEL.F_ATT:
            SPACE.update({'MODEL.R': hp.choice('R', [1, 2, 4])})
            INT_KEYS.append('MODEL.R')

        if cfg.MODEL.BRICS and cfg.LOSS.CL_LOSS:
            SPACE.update({'LOSS.ALPHA': hp.choice('alpha', [0.1, 0.15, 0.2, 0.25])})
            SPACE.update({'LOSS.TEMPERATURE': hp.choice('temperature', [0.07, 0.1, 0.2])})

        if cfg.HYPER_REMOVE is not None:
            for i in cfg.HYPER_REMOVE:
                del SPACE[i]
            INT_KEYS = [i for i in INT_KEYS if i not in cfg.HYPER_REMOVE]
        hyperopt(cfg, logger)

    else:
        logger.info(cfg.dump())
        print('The', cfg.DATA.DATASET, 'task:', cross_validate(cfg, logger))

