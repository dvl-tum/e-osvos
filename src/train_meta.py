import os
import socket
import timeit
from datetime import datetime

from itertools import chain
import networks.vgg_osvos as vo
from networks.drn_seg import DRNSeg
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
                                 dict_to_html)
from pytorch_tools.ingredients import (get_device,
                                       set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import LineVis, TextVis
from pytorch_tools.data import EpochSampler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from util.helper_func import run_loader, train_val


ex = sacred.Experiment('osvos-meta', ingredients=[torch_ingredient])
ex.add_config('cfgs/meta.yaml')
train_val = ex.capture(train_val)


@ex.capture
def init_vis(db_train, env_suffix, _config, _run, torch_cfg):
    vis_dict = {}
    run_name = f"{_run.experiment_info['name']}_{env_suffix}"

    opts = dict(
        title="OSVOS META",
        xlabel='NUM META RUNS',
        width=750,
        height=300,
        legend=['MEAN seq RUN TRAIN loss',
                # 'RUN BPTT ITER loss',
                # 'RUN BPTT loss',
                'MIN seq META loss',
                'MAX seq META loss',
                'MEAN seq META loss',
                'RUN TIME'])
    vis_dict['meta_metrics_vis'] = LineVis(opts, env=run_name, **torch_cfg['vis'])

    opts = dict(title="CONFIG and NON META BASELINE", width=300, height=1000)
    vis_dict['config_vis'] = TextVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['config_vis'].plot(dict_to_html(_config))

    for seq_name in db_train.seqs_dict.keys():
        opts = dict(
            title=f"{seq_name} - MODEL METRICS",
            xlabel='EPOCHS',
            width=750,
            height=300,
            legend=["TRAIN loss", 'BPTT ITER loss', "LR MEAN", "LR STD"],)
        vis_dict[f"{seq_name}_model_metrics"] = LineVis(
            opts, env=run_name,  **torch_cfg['vis'])
    return vis_dict


@ex.automain
def main(num_meta_runs, num_bptt_steps, meta_optim_lr, num_epochs,
         num_ave_grad, meta_optim_grad_clip, meta_optimizer_cfg, save_dir,
         run_non_meta_baseline, vis_interval, data_cfg, torch_cfg, _run, seed,
         _log, _config):
    device = get_device()
    meta_device = torch.device('cuda:1')
    set_random_seeds(seed)

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    #
    # Model
    #
    parent_state_dict = torch.load(
        os.path.join(save_dir, 'DRN_D_22', f"DRN_D_22_epoch-54.pth"),
        map_location=lambda storage, loc: storage)
    model = DRNSeg('DRN_D_22', 1, pretrained=True)
    # model = vo.OSVOS(pretrained=0)
    # parent_state_dict = torch.load(
    #     os.path.join(save_dir, 'VGG_epoch-240.pth'),
    #     map_location=lambda storage, loc: storage)
    model.load_state_dict(parent_state_dict)
    model.to(device)

    #
    # Data
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

    db_meta = db.DAVIS2016(seqs=data_cfg['seq_name'],
                           frame_id=-1,
                           transform=custom_transforms.ToTensor())
    meta_loader = DataLoader(
        db_meta,
        shuffle=data_cfg['shuffles']['meta'],
        batch_size=data_cfg['batch_sizes']['meta'],
        num_workers=2)

    #
    # visualization
    #
    if vis_interval is not None:
        vis_dict = init_vis(db_train)  # pylint: disable=E1120

    #
    # non-meta baseline
    #
    if run_non_meta_baseline:
        # lr = 1e-8
        # wd = 0.0002
        # optimizer = optim.SGD([
        #     {'params': [pr[1] for pr in model.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
        #     {'params': [pr[1] for pr in model.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
        #     {'params': [pr[1] for pr in model.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
        #     {'params': [pr[1] for pr in model.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr*2},
        #     {'params': [pr[1] for pr in model.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        #     {'params': [pr[1] for pr in model.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        #     {'params': model.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
        #     {'params': model.fuse.bias, 'lr': 2*lr/100},
        #     ], lr=lr, momentum=0.9)

        optimizer = SGDFixed(model.parameters(),
                            lr=meta_optimizer_cfg['lr_range'][1])

        run_train_losses = []
        init_meta_losses = []
        meta_losses = []
        for seq_name in db_train.seqs_dict.keys():
            db_train.set_seq(seq_name)
            db_meta.set_seq(seq_name)

            model.load_state_dict(parent_state_dict)
            model.to(device)
            model.zero_grad()

            with torch.no_grad():
                init_meta_loss = run_loader(model, meta_loader)
                init_meta_losses.append(init_meta_loss)

            run_train_loss, per_epoch_meta_losses = train_val(  # pylint: disable=E1120
                model, train_loader, meta_loader, optimizer, _log=None)
            run_train_losses.append(run_train_loss)
            meta_losses.append(per_epoch_meta_losses)

        run_train_losses = torch.tensor(run_train_losses)
        init_meta_losses = torch.tensor(init_meta_losses)
        meta_losses = torch.tensor(meta_losses)

        best_mean_meta_loss_epoch = meta_losses.mean(dim=0).argmin()

        non_meta_baseline_results_str = ("<p>RUN TRAIN loss:<br>"
            f"&nbsp;&nbsp;MIN seq: {run_train_losses.min():.2f}<br>"
            f"&nbsp;&nbsp;MAX seq: {run_train_losses.max():.2f}<br>"
            f"&nbsp;&nbsp;MEAN: {run_train_losses.mean():.2f}<br>"
            "<br>"
            "INIT META loss:<br>"
            f"&nbsp;&nbsp;MEAN seq: {init_meta_losses.mean():.2f}<br>"
            "<br>"
            "LAST META loss:<br>"
            f"&nbsp;&nbsp;MIN seq: {meta_losses[..., -1].min():.2f}<br>"
            f"&nbsp;&nbsp;MAX seq: {meta_losses[..., -1].max():.2f}<br>"
            f"&nbsp;&nbsp;MEAN: {meta_losses[..., -1].mean():.2f}</p>"
            "<br>"
            "BEST META loss:<br>"
            f"&nbsp;&nbsp;EPOCH: {best_mean_meta_loss_epoch + 1}<br>"
            f"&nbsp;&nbsp;MIN seq: {meta_losses[..., best_mean_meta_loss_epoch].min():.2f}<br>"
            f"&nbsp;&nbsp;MAX seq: {meta_losses[..., best_mean_meta_loss_epoch].max():.2f}<br>"
            f"&nbsp;&nbsp;MEAN: {meta_losses[..., best_mean_meta_loss_epoch].mean():.2f}</p>")

        if vis_interval is None:
            print(non_meta_baseline_results_str)
        else:
            vis_dict['config_vis'].plot(dict_to_html(
                _config) + non_meta_baseline_results_str)

    #
    # Meta model
    #
    meta_model = DRNSeg('DRN_D_22', 1, pretrained=True)
    meta_model.load_state_dict(parent_state_dict)
    meta_model.to(meta_device)

    meta_optim = MetaOptimizer(model, meta_model, **meta_optimizer_cfg)
    meta_optim.to(meta_device)

    meta_optim_optim = torch.optim.Adam(meta_optim.parameters(), lr=meta_optim_lr)

    meta_loss_seqs = {}
    run_train_loss_seqs = {}
    for i in range(num_meta_runs):
        start_time = timeit.default_timer()

        # db_train.set_random_seq()
        db_train.set_next_seq()
        db_meta.set_seq(db_train.seqs)
        vis_dict[f"{db_train.seqs}_model_metrics"].reset()

        model.load_state_dict(parent_state_dict)
        model.to(device)
        model.zero_grad()

        bptt_loss = ave_grad = 0
        stop_epoch = stop_train = False
        prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
        meta_optim.reset()

        # one epoch corresponds to one random transformed first frame instance
        run_train_loss = []
        for epoch in range(0, num_epochs * num_ave_grad):
            set_random_seeds(seed + epoch)# + i)
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

                    meta_model, stop_epoch = meta_optim.step(
                        db_train.get_seq_id() + 1)
                    model.zero_grad()

                    # db_meta.set_random_frame_id()

                    bptt_iter_loss = 0.0
                    for meta_batch in meta_loader:
                        meta_inputs, meta_gts = meta_batch['image'], meta_batch['gt']
                        meta_inputs, meta_gts = meta_inputs.to(
                            meta_device), meta_gts.to(meta_device)

                        meta_outputs = meta_model(meta_inputs)

                        bptt_iter_loss = class_balanced_cross_entropy_loss(
                            meta_outputs[-1], meta_gts, size_average=False)

                    # run_bptt_iter_loss.append(bptt_iter_loss.item())
                    if vis_interval is not None:
                        lr  = meta_optim.state['log_lr'].exp()
                        vis_dict[f"{db_train.seqs}_model_metrics"].plot(
                            [train_loss, bptt_iter_loss, lr.mean(), lr.std()], epoch)

                    bptt_loss += bptt_iter_loss - prev_bptt_iter_loss
                    prev_bptt_iter_loss = bptt_iter_loss.detach()

                    # Update the parameters of the meta optimizer
                    if ((epoch + 1) / num_ave_grad) % num_bptt_steps == 0:
                        # run_bptt_loss.append(bptt_loss.item())
                        meta_optim.zero_grad()
                        bptt_loss.backward()

                        # for param in meta_optim.parameters():
                        #     param.grad.clamp_(-1.0 * meta_optim_grad_clip,
                        #                       meta_optim_grad_clip)
                        meta_optim_optim.step()

                        meta_optim.reset(keep_state=True)
                        prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
                        bptt_loss = 0

                        if stop_epoch:
                            stop_train = True
                            break

            if stop_train:
                break

        run_train_loss_seqs[db_train.seqs] = torch.tensor(
            run_train_loss).mean()

        with torch.no_grad():
            meta_loss = run_loader(model, meta_loader)
            meta_loss_seqs[db_train.seqs] = meta_loss.item()

        stop_time = timeit.default_timer()
        if vis_interval is not None and not i % vis_interval:
            meta_metrics = [torch.tensor(list(run_train_loss_seqs.values())).mean(),
                            torch.tensor(list(meta_loss_seqs.values())).min(),
                            torch.tensor(list(meta_loss_seqs.values())).max(),
                            torch.tensor(list(meta_loss_seqs.values())).mean(),
                            stop_time - start_time]
            vis_dict['meta_metrics_vis'].plot(meta_metrics, i)
