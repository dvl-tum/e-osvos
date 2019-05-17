import os
import socket
import timeit
from datetime import datetime

import imageio
from itertools import chain
import networks.vgg_osvos as vo
from networks.drn_seg import DRNSeg
from networks.drn import model_zoo, model_urls
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
from meta_stopping.utils import (compute_loss,
                                 flat_grads_from_model,
                                 flat_mean_grads_from_model,
                                 flat_std_grads_from_model)
from pytorch_tools.ingredients import (MONGODB_PORT, get_device,
                                       load_model_from_db, print_config,
                                       save_model_to_db, save_model_to_path,
                                       set_random_seeds, torch_ingredient)
from pytorch_tools.vis import LineVis
from pytorch_tools.data import EpochSampler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from util import visualize as viz


ex = sacred.Experiment('osvos_meta', ingredients=[torch_ingredient])
ex.add_config('config.yaml')


@ex.capture
def train_online(model, train_loader, num_epochs, num_ave_grad, save_dir,
                 meta_optimizer_cfg, data_cfg, _log, seed):

    device = get_device()
    num_epochs = num_epochs * num_ave_grad

    # Use the following optimizer
    # lr = 1e-8
    # momentum = 0.9
    # wd = 0.0002
    wd = 0.0
    momentum = 0.0
    # optimizer = optim.SGD([
    #     {'params': [pr for na, pr in model.stages.named_parameters(
    #     ) if 'weight' in na], 'weight_decay': wd},
    #     # {'params': [pr for na, pr in model.stages.named_parameters()
    #     #             if 'bias' in na], 'lr': lr * 2},
    #     {'params': [pr for na, pr in model.stages.named_parameters()
    #                 if 'bias' in na], 'lr': lr},
    #     {'params': [pr for na, pr in model.side_prep.named_parameters(
    #     ) if 'weight' in na], 'weight_decay': wd},
    #     # {'params': [pr for na, pr in model.side_prep.named_parameters()
    #     #             if 'bias' in na], 'lr': lr*2},
    #     {'params': [pr for na, pr in model.side_prep.named_parameters()
    #                 if 'bias' in na], 'lr': lr},
    #     {'params': [pr for na, pr in model.upscale.named_parameters()
    #                 if 'weight' in na], 'lr': 0},
    #     {'params': [pr for na, pr in model.upscale_.named_parameters()
    #                 if 'weight' in na], 'lr': 0},
    #     # {'params': model.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
    #     # {'params': model.fuse.bias, 'lr': 2*lr/100},
    #     {'params': model.fuse.weight, 'lr': lr, 'weight_decay': wd},
    #     # {'params': model.fuse.bias, 'lr': 2*lr},
    #     {'params': model.fuse.bias, 'lr': lr},
    # ], lr=lr, momentum=momentum)

    optimizer = optim.SGD(model.parameters(),
                          lr=meta_optimizer_cfg['lr_range'][1],
                          momentum=momentum)

    loss_tr = []
    ave_grad = 0

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs / 2, gamma=0.1)

    _log.info(f"Train regular OSVOS - SEQUENCE: {data_cfg['seq_name']}")
    # start_time = timeit.default_timer()
    for epoch in range(0, num_epochs):
        # One training epoch
        running_loss_tr = 0
        set_random_seeds(seed + epoch)
        for _, sample_batched in enumerate(train_loader):
            inputs, gts = sample_batched['image'], sample_batched['gt']
            inputs, gts = inputs.to(device), gts.to(device)

            outputs = model.forward(inputs)

            # Compute the fuse loss
            loss = class_balanced_cross_entropy_loss(
                outputs[-1], gts, size_average=False)
            running_loss_tr += loss.item()

            # log
            if epoch % (num_epochs // 20) == (num_epochs // 20 - 1):
                running_loss_tr /= len(train_loader)
                loss_tr.append(running_loss_tr)

                _log.info(f'[epoch: {epoch + 1} loss: {running_loss_tr:.2f}]')

            loss /= num_ave_grad
            loss.backward()
            ave_grad += 1

            # Update the weights once in num_ave_grad forward passes
            if ave_grad % num_ave_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
                ave_grad = 0

                # scheduler.step()


@ex.capture
def test(model, test_loader, save_dir, data_cfg, _log):
    if save_dir is not None:
        save_dir_res = os.path.join(save_dir, 'Results', data_cfg['seq_name'])
        if not os.path.exists(save_dir_res):
            os.makedirs(save_dir_res)

    device = get_device()
    if _log is not  None:
        _log.info('Testing Network')
    with torch.no_grad():
        # Main Testing Loop
        running_loss_tr = 0
        for _, sample_batched in enumerate(test_loader):

            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']
            inputs, gts = img.to(device), gt.to(device)
            outputs = model.forward(inputs)

            loss = class_balanced_cross_entropy_loss(
                outputs[-1], gts, size_average=False)
            running_loss_tr += loss.item()

            if save_dir is not None:
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
    if _log is not  None:
        _log.info(f"Mean test loss: {test_loss:.2f}")
    return test_loss


@ex.automain
def main(num_meta_runs, num_bptt_steps, meta_optim_lr, num_epochs,
         num_ave_grad, meta_optim_grad_clip, meta_optimizer_cfg, save_dir,
         run_non_meta_baseline,
         vis_interval, data_cfg, torch_cfg, _run, seed, _log):
    device = get_device()
    meta_device = torch.device('cuda:1')
    set_random_seeds(seed)
    run_name = f"{_run._id}_{_run.experiment_info['name']}"

    if vis_interval is not None:
        metrics_opts = dict(
            title=f"OSVOS META",
            xlabel='NUM META RUNS',
            width=750,
            height=300,
            legend=['RUN TRAIN loss', 'RUN BPTT ITER loss', 'RUN BPTT loss', 'TEST loss', 'RUN TIME'])
        metrics_vis = LineVis(metrics_opts, env=run_name, **torch_cfg['vis'])

        model_metrics_opts = dict(
            title=f"MODEL METRICS",
            xlabel='EPOCHS',
            ylabel='LOSS/ACCURACY',
            width=750,
            height=300,
            legend=["TRAIN loss", 'BPTT ITER loss', "LR MEAN", "LR STD"],)
        model_metrics_vis = LineVis(
            model_metrics_opts, env=run_name,  **torch_cfg['vis'])

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    # Model
    # parent_state_dict = torch.load(os.path.join(save_dir, f"parent_epoch-239.pth"),
    #                                map_location=lambda storage, loc: storage)
    parent_state_dict = torch.load(os.path.join(save_dir, 'DRN_D_22', f"DRN_D_22_epoch-59.pth"),
                                   map_location=lambda storage, loc: storage)

    model = DRNSeg('DRN_D_22', 1, pretrained=True)
    # model = vo.OSVOS(pretrained=0)
    model.load_state_dict(parent_state_dict)
    # no_grad_param_iter = chain(model.upscale.parameters(),
    #                            model.upscale_.parameters(),
    #                            model.score_dsn.parameters())
    # for p in no_grad_param_iter:
    #     p.requires_grad_(False)
    model.to(device)

    # Data
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.ScaleNRotate(
                                                rots=(-30, 30), scales=(.75, 1.25)),
                                              tr.ToTensor()])

    db_train = db.DAVIS2016(train=True, db_root_dir='./data/DAVIS-2016',
                            transform=composed_transforms, seq_name=data_cfg['seq_name'])
    batch_sampler = EpochSampler(db_train, data_cfg['shuffles']['train'], data_cfg['batch_sizes']['train'])
    train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2)

    db_test = db.DAVIS2016(train=False, db_root_dir='./data/DAVIS-2016',
                           transform=tr.ToTensor(), seq_name=data_cfg['seq_name'])
    test_loader = DataLoader(
        db_test,
        batch_size=data_cfg['batch_sizes']['test'],
        shuffle=False,
        num_workers=2)
    # dataloader for meta test loss
    test_train_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)

    if run_non_meta_baseline:
        train_online(model, train_loader)  # pylint: disable=E1120
        test(model, test_loader)  # pylint: disable=E1120
        model.load_state_dict(parent_state_dict)
        model.to(device)
        model.zero_grad()

    # Meta model
    meta_model = DRNSeg('DRN_D_22', 1, pretrained=True)
    # meta_model = vo.OSVOS(pretrained=0)
    meta_model.load_state_dict(parent_state_dict)
    # no_grad_param_iter = chain(meta_model.upscale.parameters(),
    #                            meta_model.upscale_.parameters(),
    #                            meta_model.score_dsn.parameters())
    # for p in no_grad_param_iter:
    #     p.requires_grad_(False)
    meta_model.to(meta_device)

    meta_optim = MetaOptimizer(meta_model, **meta_optimizer_cfg)
    meta_optim.to(meta_device)

    meta_optim_optim = torch.optim.Adam(meta_optim.parameters(), lr=meta_optim_lr)

    num_epochs = num_epochs * num_ave_grad

    for i in range(num_meta_runs):
        start_time = timeit.default_timer()
        # model = vo.OSVOS(pretrained=0)
        model.load_state_dict(parent_state_dict)
        # no_grad_param_iter = chain(model.upscale.parameters(),
        #                         model.upscale_.parameters(),
        #                         model.score_dsn.parameters())
        # for p in no_grad_param_iter:
        #     p.requires_grad_(False)
        model.to(device)
        model.zero_grad()
        model_metrics_vis.reset()

        bptt_loss = ave_grad = 0
        stop_epoch = stop_train = False
        prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
        meta_optim.reset_lstm(keep_state=False, model=model)

        # one epoch corresponds to one random transformed first frame instance
        run_train_loss = []
        run_bptt_iter_loss = []
        run_bptt_loss = []
        for epoch in range(0, num_epochs):
            set_random_seeds(seed + epoch)# + num_meta_runs)
            for train_batch in train_loader:
                # train
                train_inputs, train_gts = train_batch['image'], train_batch['gt']
                train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

                train_outputs = model(train_inputs)

                train_loss = class_balanced_cross_entropy_loss(
                    train_outputs[-1], train_gts, size_average=False)
                train_loss /= num_ave_grad
                train_loss.backward()
                run_train_loss.append(train_loss.item())

                ave_grad += 1

                # Update the weights once in num_ave_grad forward passes
                if ave_grad % num_ave_grad == 0:
                    ave_grad = 0

                    mean_grads = flat_mean_grads_from_model(model)
                    std_grads = flat_std_grads_from_model(model)

                    meta_model, stop_epoch = meta_optim.meta_update(
                        model, mean_grads, std_grads, train_loss.detach(), train_loss.detach())

                    model.zero_grad()

                    bptt_iter_loss = 0.0
                    # for test_batch in test_train_loader:
                    for test_batch in reversed(test_train_loader):
                        test_inputs, test_gts = test_batch['image'], test_batch['gt']
                        test_inputs, test_gts = test_inputs.to(
                            meta_device), test_gts.to(meta_device)

                        test_outputs = meta_model(test_inputs)

                        bptt_iter_loss = class_balanced_cross_entropy_loss(
                            test_outputs[-1], test_gts, size_average=False)
                        break

                    # train_inputs, train_gts = train_inputs.to(
                    #     meta_device), train_gts.to(meta_device)
                    # test_outputs = meta_model(train_inputs)
                    # bptt_iter_loss = class_balanced_cross_entropy_loss(
                    #     test_outputs[-1], train_gts, size_average=False)

                    run_bptt_iter_loss.append(bptt_iter_loss.item())
                    lr  = meta_optim.state['log_lr'].exp()
                    model_metrics_vis.plot(
                        [train_loss, bptt_iter_loss, lr.mean(), lr.std()], epoch)

                    bptt_loss += bptt_iter_loss - prev_bptt_iter_loss
                    prev_bptt_iter_loss = bptt_iter_loss.detach()

                    # Update the parameters of the meta optimizer
                    if ((epoch + 1) / num_ave_grad) % num_bptt_steps == 0:
                        run_bptt_loss.append(bptt_loss.item())
                        meta_optim.zero_grad()
                        bptt_loss.backward()

                        for param in meta_optim.parameters():
                            param.grad.clamp_(-1.0 * meta_optim_grad_clip,
                                              meta_optim_grad_clip)
                        meta_optim_optim.step()

                        meta_optim.reset_lstm(keep_state=True, model=model)
                        prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
                        bptt_loss = 0

                        if stop_epoch:
                            stop_train = True
                            break

            if stop_train:
                break

        test_loss = test(model, test_loader, save_dir=None, _log=None)  # pylint: disable=E1120
        stop_time = timeit.default_timer()
        if vis_interval is not None and not i % vis_interval:
            metrics_vis.plot([torch.tensor(run_train_loss).mean(),
                              torch.tensor(run_bptt_iter_loss).mean(),
                              torch.tensor(run_bptt_loss).mean(),
                              test_loss,
                              stop_time - start_time], i)

