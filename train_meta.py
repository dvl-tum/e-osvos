import os
import socket
import timeit
from datetime import datetime

import imageio
import networks.vgg_osvos as vo
import sacred
import torch
import torch.optim as optim
from dataloaders import custom_transforms as tr
from dataloaders import davis_2016 as db
from dataloaders.helpers import *
from layers.osvos_layers import class_balanced_cross_entropy_loss
from meta_stopping.data import init_data_loaders
from meta_stopping.meta_optim import MetaOptimizer
from meta_stopping.model import Model
from meta_stopping.utils import compute_loss, flat_grads_from_model
from pytorch_tools.ingredients import (MONGODB_PORT, get_device,
                                       load_model_from_db, print_config,
                                       save_model_to_db, save_model_to_path,
                                       set_random_seeds, torch_ingredient)
from pytorch_tools.vis import LineVis
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from util import visualize as viz

ex = sacred.Experiment('osvos_meta', ingredients=[torch_ingredient])
ex.add_config('config.yaml')


@ex.capture
def train_online(model, train_loader, num_epochs, num_ave_grad, save_dir,
                 data_cfg, _log, _seed):
    # log tensorboard
    # log_dir = os.path.join(save_dir, 'runs',
    #     f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{socket.gethostname()}-{data_cfg['seq_name']}")
    # writer = SummaryWriter(log_dir=log_dir, comment='-meta')

    device = get_device()
    num_epochs = num_epochs * num_ave_grad

    # Use the following optimizer
    lr = 1e-8
    wd = 0.0002
    optimizer = optim.SGD([
        {'params': [pr for na, pr in model.stages.named_parameters(
        ) if 'weight' in na], 'weight_decay': wd},
        {'params': [pr for na, pr in model.stages.named_parameters()
                    if 'bias' in na], 'lr': lr * 2},
        {'params': [pr for na, pr in model.side_prep.named_parameters(
        ) if 'weight' in na], 'weight_decay': wd},
        {'params': [pr for na, pr in model.side_prep.named_parameters()
                    if 'bias' in na], 'lr': lr*2},
        {'params': [pr for na, pr in model.upscale.named_parameters()
                    if 'weight' in na], 'lr': 0},
        # {'params': [pr for na, pr in model.upscale_.named_parameters()
        #             if 'weight' in na], 'lr': 0},
        {'params': model.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
        {'params': model.fuse.bias, 'lr': 2*lr/100},
    ], lr=lr, momentum=0.9)

    loss_tr = []
    ave_grad = 0

    _log.info(f"Train regular OSVOS, sequence: {data_cfg['seq_name']}")
    # start_time = timeit.default_timer()

    for epoch in range(0, num_epochs):
        # One training epoch
        running_loss_tr = 0
        set_random_seeds(_seed + epoch)
        for ii, sample_batched in enumerate(train_loader):

            inputs, gts = sample_batched['image'], sample_batched['gt']

            # Forward-Backward of the mini-batch
            inputs.requires_grad_()
            inputs, gts = inputs.to(device), gts.to(device)

            outputs = model.forward(inputs)

            # Compute the fuse loss
            loss = class_balanced_cross_entropy_loss(
                outputs[-1], gts, size_average=False)
            running_loss_tr += loss.item()

            # log
            if epoch % (num_epochs//20) == (num_epochs//20 - 1):
                running_loss_tr /= len(train_loader)
                loss_tr.append(running_loss_tr)

                _log.info(
                    f'[epoch: {epoch + 1} loss: {running_loss_tr}]')
                # writer.add_scalar('data/total_loss_epoch',
                #                   running_loss_tr, epoch)

            loss /= num_ave_grad
            loss.backward()
            ave_grad += 1

            # Update the weights once in num_ave_grad forward passes
            if ave_grad % num_ave_grad == 0:
                # writer.add_scalar('data/total_loss_iter',
                #                   loss.item(), ii + len(train_loader) * epoch)
                optimizer.step()
                optimizer.zero_grad()
                ave_grad = 0

    # stop_time = timeit.default_timer()
    # _log.info(f"Online training time: {stop_time - start_time}")

    # writer.close()

@ex.capture
def test(model, test_loader, save_dir, data_cfg, _log):
    save_dir_res = os.path.join(save_dir, 'Results', data_cfg['seq_name'])
    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)

    device = get_device()
    _log.info('Testing Network')
    with torch.no_grad():
        # Main Testing Loop
        running_loss_tr = 0
        for ii, sample_batched in enumerate(test_loader):

            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']
            inputs, gts = img.to(device), gt.to(device)

            outputs = model.forward(inputs)

            loss = class_balanced_cross_entropy_loss(
                outputs[-1], gts, size_average=False)
            running_loss_tr += loss.item()

            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(
                    outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = 255 * np.squeeze(pred)
                pred = pred.astype(np.uint8)

                # Save the result, attention to the index jj
                imageio.imsave(os.path.join(
                    save_dir_res, os.path.basename(fname[jj]) + '.png'), pred)

    test_loss = running_loss_tr / len(test_loader)
    _log.info(f"Mean test loss: {test_loss}")
    return test_loss


@ex.automain
def main(num_meta_runs, num_bptt_steps, meta_optim_lr, num_epochs,
         num_ave_grad, meta_optim_grad_clip, meta_optimizer_cfg, save_dir,
         vis_interval, data_cfg, torch_cfg, _run, _seed, _log):
    device = get_device()
    meta_device = get_device()
    set_random_seeds(_seed)
    run_name = f"{_run._id}_{_run.experiment_info['name']}"

    if vis_interval is not None:
        metrics_opts = dict(
            title=f"OSVOS META",
            xlabel='NUM META RUNS',
            width=750,
            height=300,
            legend=['TEST loss'])
        metrics_vis = LineVis(metrics_opts, env=run_name,
                              port=torch_cfg['vis_port'])

        model_metrics_opts = dict(
            title=f"MODEL METRICS",
            xlabel='EPOCHS',
            ylabel='LOSS/ACCURACY',
            width=750,
            height=300,
            legend=["TRAIN LOSS", "TEST LOSS", "LR MEAN", "LR STD"],)
        model_metrics_vis = LineVis(model_metrics_opts, env=run_name,
                                    port=torch_cfg['vis_port'])

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    # Model
    parent_state_dict = torch.load(os.path.join(save_dir, f"parent_epoch-239.pth"),
                                   map_location=lambda storage, loc: storage)
    parent_state_dict = {k:v for k, v in parent_state_dict.items()
                         if 'upscale_' not in k and 'score_dsn' not in k}

    model = vo.OSVOS(pretrained=0)
    model.load_state_dict(parent_state_dict)
    model.to(device)
    for p in model.upscale.parameters():
        p.requires_grad_(False)

    # Data
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.ScaleNRotate(
                                                rots=(-30, 30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    db_train = db.DAVIS2016(train=True, db_root_dir='./data/DAVIS-2016',
                            transform=composed_transforms, seq_name=data_cfg['seq_name'])
    train_loader = DataLoader(
        db_train, batch_size=1, shuffle=data_cfg['shuffles']['train'], num_workers=1)

    db_test = db.DAVIS2016(train=False, db_root_dir='./data/DAVIS-2016',
                           transform=tr.ToTensor(), seq_name=data_cfg['seq_name'])
    test_loader = DataLoader(
        db_test, batch_size=1, shuffle=data_cfg['shuffles']['test'], num_workers=1)

    # train_online(model, train_loader)  # pylint: disable=E1120
    # test(model, test_loader)  # pylint: disable=E1120
    # model.load_state_dict(parent_state_dict)
    # model.to(device)

    # Meta model
    meta_model = vo.OSVOS(pretrained=0)
    meta_model.load_state_dict(parent_state_dict)
    meta_model.to(device)
    for p in meta_model.upscale.parameters():
        p.requires_grad_(False)

    meta_optim = MetaOptimizer(meta_model, **meta_optimizer_cfg)
    meta_optim.to(meta_device)

    meta_optim_optim = torch.optim.Adam(
        meta_optim.parameters(), lr=meta_optim_lr)

    num_epochs = num_epochs * num_ave_grad

    for i in range(num_meta_runs):
        model.load_state_dict(parent_state_dict)
        model.to(device)

        model_metrics_vis.reset()

        bptt_loss = ave_grad = 0
        stop_train = False
        prev_bptt_iter_loss = torch.zeros(1).to(device)
        meta_optim.reset_lstm(keep_state=False, model=model)

        # one epoch corresponds to one random transformed first frame instance
        for epoch in range(0, num_epochs):
            set_random_seeds(_seed + epoch)
            for train_batch in train_loader:
                # train
                train_inputs, train_gts = train_batch['image'], train_batch['gt']
                train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

                train_outputs = model(train_inputs)

                train_loss = class_balanced_cross_entropy_loss(
                    train_outputs[-1], train_gts, size_average=False)

                train_loss /= num_ave_grad
                train_loss.backward()
                ave_grad += 1

                # Update the weights once in num_ave_grad forward passes
                if ave_grad % num_ave_grad == 0:
                    ave_grad = 0

                    train_grads = flat_grads_from_model(model)

                    meta_model, stop_train = meta_optim.meta_update(
                        model, train_grads, train_grads, train_loss.detach(), train_loss.detach())

                    model.zero_grad()

                    bptt_iter_loss = 0.0
                    for test_batch in test_loader:
                        test_inputs, test_gts = test_batch['image'], test_batch['gt']
                        test_inputs, test_gts = test_inputs.to(
                            device), test_gts.to(device)

                        test_outputs = meta_model(test_inputs)

                        bptt_iter_loss = class_balanced_cross_entropy_loss(
                            test_outputs[-1], test_gts, size_average=False)
                        break

                    # TODO: train_loss and bptt_iter_loss are from different models
                    lr  = meta_optim.state['log_lr'].exp()
                    model_metrics_vis.plot(
                        [train_loss, bptt_iter_loss, lr.mean(), lr.std()], epoch)

                    if stop_train:
                        break

                    bptt_loss += bptt_iter_loss - prev_bptt_iter_loss
                    prev_bptt_iter_loss = bptt_iter_loss.detach()

                    # Update the parameters of the meta optimizer
                    if ((epoch + 1) / num_ave_grad + 1) % num_bptt_steps == 0:
                        meta_optim.zero_grad()
                        bptt_loss.backward()
                        for n, param in meta_optim.named_parameters():
                            param.grad.clamp_(-1.0 * meta_optim_grad_clip,
                                              meta_optim_grad_clip)
                        meta_optim_optim.step()

                        meta_optim.reset_lstm(keep_state=True, model=model)
                        bptt_loss = 0
                        prev_bptt_iter_loss = torch.zeros(1).to(device)

                if stop_train:
                    break

        test_loss = test(model, test_loader)
        if vis_interval is not None and not i % vis_interval:
            metrics_vis.plot([test_loss], i)
