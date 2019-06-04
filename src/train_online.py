# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime

import imageio
import sacred
import torch
from layers.osvos_layers import class_balanced_cross_entropy_loss
from meta_stopping.utils import dict_to_html
from pytorch_tools.ingredients import (get_device, set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import TextVis
from tensorboardX import SummaryWriter
from util import visualize as viz
from util.helper_func import (run_loader,
                              train_val,
                              datasets_and_loaders,
                              init_parent_model,
                              early_stopping)

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
def init_optim(model, parent_model_path, optim_cfg):
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
        if 'VGG' in parent_model_path:
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
        elif 'DRN_D_22' in parent_model_path:
            optim = torch.optim.SGD(model.parameters(), lr=lr,
                                    weight_decay=wd, momentum=mom)
        else:
            raise NotImplementedError

    return optim


@ex.automain
def main(parent_model_path, num_ave_grad, seed, validate_best_epoch,
         vis_interval, save_img, _log, _config):
    device = get_device()
    set_random_seeds(seed)

    model, parent_state_dict = init_parent_model(parent_model_path)
    model.to(device)
    db_train, train_loader, db_test, test_loader = datasets_and_loaders()  # pylint: disable=E1120
    if vis_interval is not None:
        vis_dict = init_vis(db_train)  # pylint: disable=E1120

    val_loader = None
    if validate_best_epoch:
        val_loader = test_loader

    run_train_losses = []
    val_losses = []
    test_losses = []
    for seq_name in db_train.seqs_dict.keys():
        optim = init_optim(model)  # pylint: disable=E1120

        db_train.set_seq(seq_name)
        db_test.set_seq(seq_name)

        model.load_state_dict(parent_state_dict)
        model.to(device)
        model.zero_grad()

        run_train_loss, per_epoch_val_losses = train_val(  # pylint: disable=E1120
            model, train_loader, val_loader, optim, early_stopping_func=early_stopping)
        run_train_losses.append(run_train_loss)
        val_losses.append(per_epoch_val_losses)

        with torch.no_grad():
            img_save_dir = None
            if save_img:
                img_save_dir = os.path.join(
                    os.path.dirname(parent_model_path),
                    'results',
                    seq_name)
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
            test_losses.append(run_loader(model, test_loader, img_save_dir=img_save_dir))

    run_train_losses = torch.tensor(run_train_losses)
    test_losses = torch.tensor(test_losses)

    results_str = (
        "<p>RUN TRAIN loss:<br>\n"
        f"&nbsp;&nbsp;MIN seq: {run_train_losses.min():.2f}<br>\n"
        f"&nbsp;&nbsp;MAX seq: {run_train_losses.max():.2f}<br>\n"
        f"&nbsp;&nbsp;MEAN: {run_train_losses.mean():.2f}<br>\n"
        "<br>\n"
        "LAST TEST loss:<br>\n"
        f"&nbsp;&nbsp;MIN seq: {test_losses.min():.2f}<br>\n"
        f"&nbsp;&nbsp;MAX seq: {test_losses.max():.2f}<br>\n"
        f"&nbsp;&nbsp;MEAN: {test_losses.mean():.2f}</p>\n")

    if validate_best_epoch:
        val_losses = torch.tensor(val_losses)
        best_mean_val_loss_epoch = val_losses.mean(dim=0).argmin()

        results_str += (
            "<p>BEST VAL loss:<br>\n"
            f"&nbsp;&nbsp;EPOCH: {best_mean_val_loss_epoch + 1}<br>\n"
            f"&nbsp;&nbsp;MIN seq: {val_losses[..., best_mean_val_loss_epoch].min():.2f}<br>\n"
            f"&nbsp;&nbsp;MAX seq: {val_losses[..., best_mean_val_loss_epoch].max():.2f}<br>\n"
            f"&nbsp;&nbsp;MEAN: {val_losses[..., best_mean_val_loss_epoch].mean():.2f}</p>\n")

    if vis_interval is None:
        print(results_str)
    else:
        vis_dict['config_vis'].plot(dict_to_html(_config) + results_str)
