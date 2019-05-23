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
from dataloaders import custom_transforms
from dataloaders import davis_2016 as db
from dataloaders.helpers import *
from layers.osvos_layers import class_balanced_cross_entropy_loss
from meta_stopping.data import init_data_loaders
from meta_stopping.meta_optim import MetaOptimizer, SGDFixed
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
def train_online(model, train_loader, test_loader, num_epochs, num_ave_grad, save_dir,
                 meta_optimizer_cfg, data_cfg, _log, seed):

    device = get_device()

    # Use the following optimizer
    wd = 0.0
    momentum = 0.0
    optimizer = SGDFixed(model.parameters(),
                         lr=meta_optimizer_cfg['lr_range'][1],
                         momentum=momentum)

    test_losses = []
    run_train_loss = []
    ave_grad = 0

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs / 2, gamma=0.1)

    _log.info(f"Train regular OSVOS - SEQUENCE: {data_cfg['seq_name']}")
    for epoch in range(0, num_epochs * num_ave_grad):
        set_random_seeds(seed + epoch)
        for _, sample_batched in enumerate(train_loader):
            inputs, gts = sample_batched['image'], sample_batched['gt']
            inputs, gts = inputs.to(device), gts.to(device)

            outputs = model(inputs)

            # Compute the fuse loss
            loss = class_balanced_cross_entropy_loss(
                outputs[-1], gts, size_average=False)
            run_train_loss.append(loss.item())

            loss /= num_ave_grad
            loss.backward()
            ave_grad += 1

            # Update the weights once in num_ave_grad forward passes
            if ave_grad % num_ave_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
                ave_grad = 0

                test_loss = test(model, test_loader, save_dir=None, _log=None)  # pylint: disable=E1120
                test_losses.append(test_loss.item())
                # scheduler.step()

    _log.info(
        f'RUN TRAIN loss: {torch.tensor(run_train_loss).mean().item():.2f}]')
    best_test_epoch = torch.tensor(test_losses).argmin()
    best_test_loss = torch.tensor(test_losses)[best_test_epoch]
    _log.info(
        f'BEST TEST loss/epoch: {best_test_loss:.2f}/{best_test_epoch + 1}]')


@ex.capture
def test(model, test_loader, save_dir, data_cfg, _log):
    if save_dir is not None:
        save_dir_res = os.path.join(save_dir, 'Results', data_cfg['seq_name'])
        if not os.path.exists(save_dir_res):
            os.makedirs(save_dir_res)

    device = get_device()
    run_loss = []
    if _log is not  None:
        _log.info('Testing Network')
    with torch.no_grad():
        for j, sample_batched in enumerate(test_loader):
            if j + 1 == len(test_loader):
                img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']
                inputs, gts = img.to(device), gt.to(device)
                outputs = model.forward(inputs)

                loss = class_balanced_cross_entropy_loss(
                    outputs[-1], gts, size_average=False)
                run_loss.append(loss.item())

                if save_dir is not None:
                    for jj in range(int(inputs.size()[0])):
                        pred = np.transpose(
                            outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                        pred = 1 / (1 + np.exp(-pred))
                        pred = 255 * np.squeeze(pred)
                        pred = pred.astype(np.uint8)

                        imageio.imsave(os.path.join(
                            save_dir_res, os.path.basename(fname[jj]) + '.png'), pred)

    test_loss = torch.tensor(run_loss).mean()
    if _log is not  None:
        _log.info(f"MEAN TEST loss: {test_loss}")
    return test_loss


@ex.automain
def main(num_meta_runs, num_bptt_steps, meta_optim_lr, num_epochs,
         num_ave_grad, meta_optim_grad_clip, meta_optimizer_cfg, save_dir,
         run_non_meta_baseline, vis_interval, data_cfg, torch_cfg, _run, seed,
         _log):
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
    parent_state_dict = torch.load(os.path.join(save_dir, 'DRN_D_22', f"DRN_D_22_epoch-59.pth"),
                                   map_location=lambda storage, loc: storage)

    model = DRNSeg('DRN_D_22', 1, pretrained=True)
    model.load_state_dict(parent_state_dict)
    model.to(device)

    # Data
    train_transforms = []
    if data_cfg['random_train_transform']:
        train_transforms.extend([custom_transforms.RandomHorizontalFlip(),
                                 custom_transforms.ScaleNRotate(rots=(-30, 30),
                                                                scales=(.75, 1.25))])
    train_transforms.append(custom_transforms.ToTensor())
    composed_transforms = transforms.Compose(train_transforms)

    db_train = db.DAVIS2016(seqs=data_cfg['seq_name'], one_shot=True,
                            transform=composed_transforms)
    batch_sampler = EpochSampler(db_train,
                                 data_cfg['shuffles']['train'],
                                 data_cfg['batch_sizes']['train'])
    train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2)

    db_test = db.DAVIS2016(
        seqs=data_cfg['seq_name'], transform=custom_transforms.ToTensor())
    test_loader = DataLoader(
        db_test,
        batch_size=data_cfg['batch_sizes']['test'],
        shuffle=False,
        num_workers=2)
    # dataloader for meta test loss
    meta_loader = DataLoader(
        db_test,
        batch_size=data_cfg['batch_sizes']['meta'],
        shuffle=False,
        num_workers=2)

    if run_non_meta_baseline:
        train_online(model, train_loader, test_loader)  # pylint: disable=E1120
        test(model, test_loader, save_dir=None)  # pylint: disable=E1120
        model.load_state_dict(parent_state_dict)
        model.to(device)
        model.zero_grad()

    # Meta model
    meta_model = DRNSeg('DRN_D_22', 1, pretrained=True)
    meta_model.load_state_dict(parent_state_dict)
    meta_model.to(meta_device)

    meta_optim = MetaOptimizer(meta_model, **meta_optimizer_cfg)
    meta_optim.to(meta_device)

    meta_optim_optim = torch.optim.Adam(meta_optim.parameters(), lr=meta_optim_lr)

    test_loss_seqs = {}
    for i in range(num_meta_runs):
        start_time = timeit.default_timer()

        # db_train.set_random_seq()
        db_train.set_next_seq()
        db_test.set_seq(db_train.seqs)

        model.load_state_dict(parent_state_dict)
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
        for epoch in range(0, num_epochs * num_ave_grad):
            set_random_seeds(seed + epoch)
            for train_batch in train_loader:
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
                    for j, test_batch in enumerate(meta_loader):
                        # fist frame of sequence, i.e., train frame
                        # if j == 0:
                        # last frame of sequence
                        if j + 1 == len(meta_loader):
                            test_inputs, test_gts = test_batch['image'], test_batch['gt']
                            test_inputs, test_gts = test_inputs.to(
                                meta_device), test_gts.to(meta_device)

                            test_outputs = meta_model(test_inputs)

                            bptt_iter_loss = class_balanced_cross_entropy_loss(
                                test_outputs[-1], test_gts, size_average=False)
                            break

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

                        # for param in meta_optim.parameters():
                        #     param.grad.clamp_(-1.0 * meta_optim_grad_clip,
                        #                       meta_optim_grad_clip)
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
        test_loss_seqs[db_train.seqs] = test_loss.item()
        stop_time = timeit.default_timer()
        if vis_interval is not None and not i % vis_interval:
            metrics_vis.plot([torch.tensor(run_train_loss).mean(),
                              torch.tensor(run_bptt_iter_loss).mean(),
                              torch.tensor(run_bptt_loss).mean(),
                            #   test_loss,
                              torch.tensor(list(test_loss_seqs.values())).mean(),
                              stop_time - start_time], i)
