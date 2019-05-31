# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime

import imageio
import networks.vgg_osvos as vo
import sacred
import torch
import torch.optim as optim
from dataloaders import custom_transforms
from dataloaders import davis_2016 as db
from dataloaders.helpers import *
from layers.osvos_layers import class_balanced_cross_entropy_loss
from meta_stopping.utils import dict_to_html
from mypath import Path
from networks.drn_seg import DRNSeg
from pytorch_tools.data import EpochSampler
from pytorch_tools.ingredients import (get_device, set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import TextVis
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from util import visualize as viz
from util.helper_func import run_loader, train_val

torch_ingredient.add_config('cfgs/torch.yaml')
ex = sacred.Experiment('osvos-online', ingredients=[torch_ingredient])
ex.add_config('cfgs/online.yaml')
ex.add_named_config('VGG', 'cfgs/online_vgg.yaml')
train_val = ex.capture(train_val)


@ex.capture
def init_vis(db_train, env_suffix, _config, _run, torch_cfg):
    vis_dict = {}
    run_name = f"{_run.experiment_info['name']}_{env_suffix}"

    opts = dict(title="CONFIG and RESULTS", width=300, height=1000)
    vis_dict['config_vis'] = TextVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['config_vis'].plot(dict_to_html(_config))

    return vis_dict


@ex.automain
def main(parent_model_cfg, optimizer_cfg, num_ave_grad, num_epochs, seed,
         validate_best_epoch, vis_interval, save_dir, data_cfg, _log, _config):
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    device = get_device()
    set_random_seeds(seed)

    #
    # model
    #
    if parent_model_cfg['name'] == 'VGG':
        model = vo.OSVOS(pretrained=0)
    elif parent_model_cfg['name'] == 'DRN_D_22':
        model = DRNSeg('DRN_D_22', 1, pretrained=True)

    parent_state_dict = torch.load(
        os.path.join(
            save_dir, parent_model_cfg['name'],
            f"{parent_model_cfg['name']}_epoch-{parent_model_cfg['epoch']}.pth"),
        map_location=lambda storage, loc: storage)
    model.load_state_dict(parent_state_dict)
    model.to(device)

    #
    # optimizer
    #
    lr = optimizer_cfg['lr']
    wd = optimizer_cfg['wd']
    mom = optimizer_cfg['mom']
    if parent_model_cfg['name'] == 'VGG':
        optimizer = optim.SGD([
            {'params': [pr[1] for pr in model.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
            {'params': [pr[1] for pr in model.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
            {'params': [pr[1] for pr in model.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
            {'params': [pr[1] for pr in model.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr*2},
            {'params': [pr[1] for pr in model.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
            {'params': [pr[1] for pr in model.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
            {'params': model.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
            {'params': model.fuse.bias, 'lr': 2*lr/100},
            ], lr=lr, momentum=mom)
    elif parent_model_cfg['name'] == 'DRN_D_22':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              weight_decay=wd, momentum=mom)
    else:
        raise NotImplementedError

    #
    # data
    #
    train_transforms = []
    if data_cfg['random_train_transform']:
        train_transforms.extend([custom_transforms.RandomHorizontalFlip(),
                                 custom_transforms.ScaleNRotate(rots=(-30, 30),
                                                                scales=(.75, 1.25))])
    train_transforms.append(custom_transforms.ToTensor())
    composed_transforms = transforms.Compose(train_transforms)

    db_train = db.DAVIS2016(seqs=data_cfg['seq_name'],
                            frame_id=0,
                            transform=composed_transforms)
    batch_sampler = EpochSampler(db_train,
                                 data_cfg['shuffles']['train'],
                                 data_cfg['batch_sizes']['train'])
    train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2)

    db_test = db.DAVIS2016(seqs=data_cfg['seq_name'],
                           transform=custom_transforms.ToTensor())
    test_loader = DataLoader(
        db_test,
        shuffle=data_cfg['shuffles']['test'],
        batch_size=data_cfg['batch_sizes']['test'],
        num_workers=2)

    #
    # visualization
    #
    if vis_interval is not None:
        vis_dict = init_vis(db_train)  # pylint: disable=E1120

    #
    # train
    #
    val_loader = None
    if validate_best_epoch:
        val_loader = test_loader

    run_train_losses = []
    val_losses = []
    test_losses = []
    for seq_name in db_train.seqs_dict.keys():
        db_train.set_seq(seq_name)
        db_test.set_seq(seq_name)

        model.load_state_dict(parent_state_dict)
        model.to(device)
        model.zero_grad()

        run_train_loss, per_epoch_val_losses = train_val(  # pylint: disable=E1120
            model, train_loader, val_loader, optimizer)
        run_train_losses.append(run_train_loss)
        val_losses.append(per_epoch_val_losses)

        with torch.no_grad():
            save_dir_res = None
            if save_dir is not None:
                save_dir_res = os.path.join(save_dir, parent_model_cfg['name'], 'results', seq_name)
                if not os.path.exists(save_dir_res):
                    os.makedirs(save_dir_res)
            test_losses.append(run_loader(
                model, test_loader, save_dir=save_dir_res))

    run_train_losses = torch.tensor(run_train_losses)
    test_losses = torch.tensor(test_losses)

    results_str = (
        "<p>RUN TRAIN loss:<br>"
        f"&nbsp;&nbsp;MIN seq: {run_train_losses.min():.2f}<br>"
        f"&nbsp;&nbsp;MAX seq: {run_train_losses.max():.2f}<br>"
        f"&nbsp;&nbsp;MEAN: {run_train_losses.mean():.2f}<br>"
        "<br>"
        "LAST TEST loss:<br>"
        f"&nbsp;&nbsp;MIN seq: {test_losses.min():.2f}<br>"
        f"&nbsp;&nbsp;MAX seq: {test_losses.max():.2f}<br>"
        f"&nbsp;&nbsp;MEAN: {test_losses.mean():.2f}</p>")

    if validate_best_epoch:
        val_losses = torch.tensor(val_losses)
        best_mean_val_loss_epoch = val_losses.mean(dim=0).argmin()

        results_str += (
            "BEST VAL loss:<br>"
            f"&nbsp;&nbsp;EPOCH: {best_mean_val_loss_epoch + 1}<br>"
            f"&nbsp;&nbsp;MIN seq: {val_losses[..., best_mean_val_loss_epoch].min():.2f}<br>"
            f"&nbsp;&nbsp;MAX seq: {val_losses[..., best_mean_val_loss_epoch].max():.2f}<br>"
            f"&nbsp;&nbsp;MEAN: {val_losses[..., best_mean_val_loss_epoch].mean():.2f}</p>")

    if vis_interval is None:
        print(results_str)
    else:
        vis_dict['config_vis'].plot(dict_to_html(_config) + results_str)
