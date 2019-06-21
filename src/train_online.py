# Package Includes
from __future__ import division

import itertools
import os
import socket
import timeit
from datetime import datetime

import davis
import imageio
import numpy as np
import sacred
import torch
from davis import DAVISLoader, Segmentation, db_eval
from layers.osvos_layers import class_balanced_cross_entropy_loss
from meta_stopping.utils import dict_to_html
from prettytable import PrettyTable
from pytorch_tools.ingredients import (get_device, set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import TextVis
from tensorboardX import SummaryWriter
from util import visualize as viz
from util.helper_func import (datasets_and_loaders, early_stopping,
                              init_parent_model, run_loader, train_val)

torch_ingredient.add_config('cfgs/torch.yaml')
ex = sacred.Experiment('osvos-online', ingredients=[torch_ingredient])

ex.add_config('cfgs/online.yaml')
ex.add_named_config('VGG', 'cfgs/online_vgg.yaml')

train_val = ex.capture(train_val)
early_stopping = ex.capture(early_stopping, prefix='train_early_stopping')
datasets_and_loaders = ex.capture(datasets_and_loaders, prefix='data')


@ex.capture
def init_vis(db_train, env_suffix, _config, _run, torch_cfg):
    vis_dict = {}
    run_name = f"{_run.experiment_info['name']}_{env_suffix}"

    opts = dict(title="CONFIG and RESULTS", width=300, height=1000)
    vis_dict['config_vis'] = TextVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['config_vis'].plot(dict_to_html(_config))

    return vis_dict


@ex.capture
def init_optim(model, parent_model_cfg, optim_cfg):
    device = get_device()

    if optim_cfg['file_dir'] is not None:
        optim = torch.load(optim_cfg['file_dir'],
                           map_location=lambda storage, loc: storage)
        optim.to(device)
        optim.set_model(model)
        optim.set_meta_model(model)
        optim.reset()
        optim.eval()
    else:
        lr = optim_cfg['lr']
        wd = optim_cfg['wd']
        mom = optim_cfg['mom']
        if 'VGG' in parent_model_cfg['base_path']:
            optim = torch.optim.SGD([
                {'params': [pr[1] for pr in model.stages.named_parameters(
                ) if 'weight' in pr[0]], 'weight_decay': wd},
                {'params': [pr[1] for pr in model.stages.named_parameters(
                ) if 'bias' in pr[0]], 'lr': lr * 2},
                {'params': [pr[1] for pr in model.side_prep.named_parameters(
                ) if 'weight' in pr[0]], 'weight_decay': wd},
                {'params': [pr[1] for pr in model.side_prep.named_parameters(
                ) if 'bias' in pr[0]], 'lr': lr*2},
                {'params': [pr[1] for pr in model.upscale.named_parameters(
                ) if 'weight' in pr[0]], 'lr': 0},
                {'params': [pr[1] for pr in model.upscale_.named_parameters(
                ) if 'weight' in pr[0]], 'lr': 0},
                {'params': model.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
                {'params': model.fuse.bias, 'lr': 2*lr/100},
            ], lr=lr, momentum=mom)
        else:
            optim = torch.optim.SGD(model.parameters(), lr=lr,
                                    weight_decay=wd, momentum=mom)

    return optim


@ex.automain
def main(parent_model_cfg, num_ave_grad, seed, validate_best_epoch,
         vis_interval, save_img_and_eval, _log, _config, data):
    device = get_device()
    set_random_seeds(seed)

    model, parent_state_dicts = init_parent_model(parent_model_cfg)
    model.to(device)
    db_train, train_loader, db_test, test_loader = datasets_and_loaders()  # pylint: disable=E1120
    if vis_interval is not None:
        vis_dict = init_vis(db_train)  # pylint: disable=E1120

    val_loader = None
    if validate_best_epoch:
        val_loader = test_loader

    train_split_X_val_seqs = []
    for i, p in enumerate(parent_model_cfg['split_model_path']):
        if 'split' in p:
            split_file = os.path.join(f'data/DAVIS-2016/train_split_{i + 1}_val.txt')
        else:
            split_file = os.path.join('data/DAVIS-2016/test_seqs.txt')
        seqs = [s.rstrip('\n') for s in open(split_file)]
        train_split_X_val_seqs.append(seqs)

    run_train_losses = []
    val_losses = []
    test_losses = []
    init_test_losses = []
    for seq_name in db_train.seqs_dict.keys():
        optim = init_optim(model)  # pylint: disable=E1120

        db_train.set_seq(seq_name)
        db_test.set_seq(seq_name)

        for seqs_list, p_s_d in zip(train_split_X_val_seqs, parent_state_dicts):
            if seq_name in seqs_list:
                model.load_state_dict(p_s_d)
        model.to(device)
        model.zero_grad()

        init_test_losses.append(run_loader(model, test_loader))

        run_train_loss, per_epoch_val_losses = train_val(  # pylint: disable=E1120
            model, train_loader, val_loader, optim, early_stopping_func=early_stopping)
        run_train_losses.append(run_train_loss)
        val_losses.append(per_epoch_val_losses)

        with torch.no_grad():
            img_save_dir = None
            if save_img_and_eval:
                img_save_dir = os.path.join(
                    parent_model_cfg['base_path'],
                    'results',
                    seq_name)
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
            test_losses.append(run_loader(model, test_loader, img_save_dir=img_save_dir))

    run_train_losses = torch.tensor(run_train_losses)
    test_losses = torch.tensor(test_losses)
    init_test_losses = torch.tensor(init_test_losses)

    if save_img_and_eval:
        if data['seq_name'] == 'train_seqs':
            phase = davis.phase['TRAIN']
        elif data['seq_name'] == 'test_seqs':
            phase = davis.phase['VAL']
        else:
            phase = None

        if phase is not None:
            db = DAVISLoader('2016', phase, True)

            metrics = ['J', 'F']
            statistics = ['mean','recall','decay']
            results_dir = os.path.join(
                parent_model_cfg['base_path'], 'results')

            # Load segmentations
            segmentations = [Segmentation(os.path.join(results_dir, s), True)
                            for s in db.iternames()]

            # Evaluate results
            evaluation = db_eval(db, segmentations, metrics)

            # Print results
            table = PrettyTable(['Method']+[p[0]+'_'+p[1] for p in
                                            itertools.product(metrics, statistics)])

            table.add_row([os.path.basename(results_dir)] + ["%.3f" % np.round(
                evaluation['dataset'][metric][statistic] , 3) for metric, statistic
                in itertools.product(metrics, statistics)])

            for s_name, s_eval in evaluation['sequence'].items():
                print(f"{s_name} - {s_eval['J']['mean'][0]:.2f}")
            print(str(table) + "\n")

    results_str = (
        "<p>RUN TRAIN loss:<br>\n"
        f"&nbsp;&nbsp;MIN seq: {run_train_losses.min():.2f}<br>\n"
        f"&nbsp;&nbsp;MAX seq: {run_train_losses.max():.2f}<br>\n"
        f"&nbsp;&nbsp;MEAN: {run_train_losses.mean():.2f}<br>\n"
        "<br>\n"
        "INIT TEST loss:<br>\n"
        f"&nbsp;&nbsp;MIN seq: {init_test_losses.min():.2f}<br>\n"
        f"&nbsp;&nbsp;MAX seq: {init_test_losses.max():.2f}<br>\n"
        f"&nbsp;&nbsp;MEAN: {init_test_losses.mean():.2f}<br>\n"
        "<br>\n"
        "LAST TEST loss:<br>\n"
        f"&nbsp;&nbsp;MIN seq: {test_losses.min():.2f}<br>\n"
        f"&nbsp;&nbsp;MAX seq: {test_losses.max():.2f}<br>\n"
        f"&nbsp;&nbsp;MEAN: {test_losses.mean():.2f}</p>\n"
        )

    if validate_best_epoch:
        val_losses = torch.tensor(val_losses)
        best_mean_val_loss_epoch = val_losses.mean(dim=0).argmin()

        results_str += (
            "<p>BEST VAL loss:<br>\n"
            f"&nbsp;&nbsp;EPOCH: {best_mean_val_loss_epoch + 1}<br>\n"
            f"&nbsp;&nbsp;MIN seq: {val_losses[..., best_mean_val_loss_epoch].min():.2f}<br>\n"
            f"&nbsp;&nbsp;MAX seq: {val_losses[..., best_mean_val_loss_epoch].max():.2f}<br>\n"
            f"&nbsp;&nbsp;MEAN: {val_losses[..., best_mean_val_loss_epoch].mean():.2f}</p>\n")

    results_str += "<p>SEQUENCES:<br>\n"
    for i, seq_name in enumerate(db_train.seqs_dict.keys()):
        results_str += (f"{seq_name}<br>\n"
                        f"RUN TRAIN loss: {run_train_losses[i]:.2f}<br>\n"
                        f"INIT TEST loss: {init_test_losses[i]:.2f}<br>\n"
                        f"LAST TEST loss: {test_losses[i]:.2f}</p>\n")

    if vis_interval is None:
        print(results_str)
    else:
        vis_dict['config_vis'].plot(dict_to_html(_config) + results_str)
