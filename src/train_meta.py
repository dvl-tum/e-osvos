import copy
import os
import socket
import timeit
from datetime import datetime
from itertools import count

import networks.vgg_osvos as vo
import sacred
import torch
from layers.osvos_layers import class_balanced_cross_entropy_loss
from meta_stopping.meta_optim import MetaOptimizer, SGDFixed
from meta_stopping.model import Model
from meta_stopping.utils import (compute_loss, dict_to_html,
                                 flat_grads_from_model)
from pytorch_tools.ingredients import (MONGODB_PORT,
                                       get_device,
                                       save_model_to_db,
                                       set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import LineVis, TextVis
from sacred.observers import MongoObserver
from tensorboardX import SummaryWriter
from util.helper_func import (run_loader,
                              train_val,
                              datasets_and_loaders,
                              init_parent_model,
                              early_stopping)

torch_ingredient.add_config('cfgs/torch.yaml')

ex = sacred.Experiment('osvos-meta', ingredients=[torch_ingredient])
ex.add_config('cfgs/meta.yaml')

# if MONGODB_PORT is not None:
#     ex.observers.append(MongoObserver.create(db_name='osvos-meta',
#                                              port=MONGODB_PORT))

MetaOptimizer = ex.capture(MetaOptimizer, prefix='meta_optim')
datasets_and_loaders = ex.capture(datasets_and_loaders, prefix='data')
early_stopping = ex.capture(early_stopping, prefix='train_early_stopping')
train_val = ex.capture(train_val)


@ex.capture
def init_vis(db_train, env_suffix, _config, _run, torch_cfg):
    run_name = f"{_run.experiment_info['name']}_{env_suffix}"
    vis_dict = {}

    opts = dict(title=f"CONFIG and NON META BASELINE (RUN: {_run._id})",
                width=300, height=1250)
    vis_dict['config_vis'] = TextVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['config_vis'].plot(dict_to_html(_config))

    legend = [
        'MEAN seq RUN TRAIN loss',
        'MIN seq META loss',
        'MAX seq META loss',
        'MEAN seq META loss',
        'NUM EPOCHS',
        'RUN TIME']
    opts = dict(
        title=f"OSVOS META  (RUN: {_run._id})",
        xlabel='NUM META RUNS',
        width=750,
        height=300,
        legend=legend)
    vis_dict['meta_metrics_vis'] = LineVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['meta_metrics_vis'].plot([0] * len(legend), 0)

    for seq_name in db_train.seqs_dict.keys():
        opts = dict(
            title=f"{seq_name} - MODEL METRICS",
            xlabel='EPOCHS',
            width=450,
            height=300,
            legend=["TRAIN loss", 'BPTT ITER loss', "LR MEAN", "LR MOM MEAN"])
        vis_dict[f"{seq_name}_model_metrics"] = LineVis(
            opts, env=run_name,  **torch_cfg['vis'])
    return vis_dict


@ex.automain
def main(bptt_cfg, num_epochs, meta_optim_optim_cfg, non_meta_baseline_cfg,
         vis_interval, torch_cfg, _run, seed, _log, _config, save_dir,
         parent_model_path):
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))
    device = get_device()
    meta_device = torch.device('cuda:1')
    set_random_seeds(seed)

    model, parent_state_dict = init_parent_model(parent_model_path)
    model.to(device)
    db_train, train_loader, db_meta, meta_loader = datasets_and_loaders()  # pylint: disable=E1120
    if vis_interval is not None:
        vis_dict = init_vis(db_train)  # pylint: disable=E1120

    if non_meta_baseline_cfg['compute']:
        run_train_loss_hist = []
        init_meta_loss_hist = []
        meta_loss_hist = []
        for seq_name in db_train.seqs_dict.keys():
            db_train.set_seq(seq_name)
            db_meta.set_seq(seq_name)

            optimizer = SGDFixed(model.parameters(), lr=non_meta_baseline_cfg['lr'])

            model.load_state_dict(parent_state_dict)
            model.to(device)
            model.zero_grad()

            with torch.no_grad():
                init_meta_loss = run_loader(model, meta_loader)
                init_meta_loss_hist.append(init_meta_loss)

            run_train_loss, per_epoch_meta_losses = train_val(  # pylint: disable=E1120
                model, train_loader, meta_loader, optimizer, non_meta_baseline_cfg['num_epochs'], _log=None)
            run_train_loss_hist.append(run_train_loss)
            meta_loss_hist.append(per_epoch_meta_losses)

        run_train_loss_hist = torch.tensor(run_train_loss_hist)
        init_meta_loss_hist = torch.tensor(init_meta_loss_hist)
        meta_loss_hist = torch.tensor(meta_loss_hist)

        best_mean_meta_loss_epoch = meta_loss_hist.mean(dim=0).argmin()

        non_meta_baseline_results_str = ("<p>RUN TRAIN loss:<br>"
            f"&nbsp;&nbsp;MIN seq: {run_train_loss_hist.min():.2f}<br>"
            f"&nbsp;&nbsp;MAX seq: {run_train_loss_hist.max():.2f}<br>"
            f"&nbsp;&nbsp;MEAN: {run_train_loss_hist.mean():.2f}<br>"
            "<br>"
            "INIT META loss:<br>"
            f"&nbsp;&nbsp;MEAN seq: {init_meta_loss_hist.mean():.2f}<br>"
            "<br>"
            "LAST META loss:<br>"
            f"&nbsp;&nbsp;MIN seq: {meta_loss_hist[..., -1].min():.2f}<br>"
            f"&nbsp;&nbsp;MAX seq: {meta_loss_hist[..., -1].max():.2f}<br>"
            f"&nbsp;&nbsp;MEAN: {meta_loss_hist[..., -1].mean():.2f}<br>"
            "<br>"
            "BEST META loss:<br>"
            f"&nbsp;&nbsp;EPOCH: {best_mean_meta_loss_epoch + 1}<br>"
            f"&nbsp;&nbsp;MIN seq: {meta_loss_hist[..., best_mean_meta_loss_epoch].min():.2f}<br>"
            f"&nbsp;&nbsp;MAX seq: {meta_loss_hist[..., best_mean_meta_loss_epoch].max():.2f}<br>"
            f"&nbsp;&nbsp;MEAN: {meta_loss_hist[..., best_mean_meta_loss_epoch].mean():.2f}</p>")

        if vis_interval is None:
            print(non_meta_baseline_results_str)
        else:
            vis_dict['config_vis'].plot(dict_to_html(
                _config) + non_meta_baseline_results_str)

    #
    # Meta model
    #
    meta_model, parent_state_dict = init_parent_model(parent_model_path)
    meta_model.to(meta_device)

    meta_optim = MetaOptimizer(model, meta_model)  # pylint: disable=E1120
    meta_optim.to(meta_device)

    meta_optim_optim = torch.optim.Adam(
        meta_optim.parameters(), lr=meta_optim_optim_cfg['lr'])

    meta_loss_seqs = {}
    run_train_loss_seqs = {}
    meta_optim_param_grad = {}
    for i in count():
        start_time = timeit.default_timer()
        meta_optim_state_dict = copy.deepcopy(meta_optim.state_dict())

        for name, param in meta_optim.named_parameters():
            meta_optim_param_grad[name] = torch.zeros_like(param)

        for seq_name in db_train.seqs_dict.keys():
            # db_train.set_random_seq()
            # db_train.set_next_seq()
            db_train.set_seq(seq_name)
            db_meta.set_seq(seq_name)
            vis_dict[f"{seq_name}_model_metrics"].reset()

            model.load_state_dict(parent_state_dict)
            model.to(device)
            model.zero_grad()

            meta_optim.load_state_dict(meta_optim_state_dict)
            meta_optim.reset()

            bptt_loss = 0
            stop_train = False
            prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
            run_train_loss_hist = []

            # one epoch corresponds to one random transformed first frame of a sequence
            if num_epochs is None:
                # epoch_iter = count()
                epoch_iter = range(bptt_cfg['epochs'] * (1 + i // bptt_cfg['runs_per_epoch_extension']))
            else:
                epoch_iter = range(num_epochs)
            for epoch in epoch_iter:
                set_random_seeds(seed + epoch)# + i)
                for train_batch in train_loader:
                    train_inputs, train_gts = train_batch['image'], train_batch['gt']
                    train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

                    train_outputs = model(train_inputs)

                    train_loss = class_balanced_cross_entropy_loss(
                        train_outputs[-1], train_gts, size_average=False)
                    train_loss.backward()
                    run_train_loss_hist.append(train_loss.item())

                    # seq_id = db_train.get_seq_id() + 1
                    meta_model, stop_train = meta_optim.step()
                    model.zero_grad()

                    stop_train = stop_train or early_stopping(  # pylint: disable=E1120
                        run_train_loss_hist)
                    # db_meta.set_random_frame_id()

                    bptt_iter_loss = 0.0
                    for meta_batch in meta_loader:
                        meta_inputs, meta_gts = meta_batch['image'], meta_batch['gt']
                        meta_inputs, meta_gts = meta_inputs.to(
                            meta_device), meta_gts.to(meta_device)

                        meta_outputs = meta_model(meta_inputs)

                        bptt_iter_loss = class_balanced_cross_entropy_loss(
                            meta_outputs[-1], meta_gts, size_average=False)

                    if vis_interval is not None:
                        lr = meta_optim.state["log_lr"].exp().mean()
                        lr_mom = meta_optim.state["lr_mom_logit"].sigmoid().mean()
                        vis_dict[f"{db_train.seqs}_model_metrics"].plot(
                            [train_loss, bptt_iter_loss, lr, lr_mom], epoch + 1)

                    bptt_loss += bptt_iter_loss - prev_bptt_iter_loss
                    prev_bptt_iter_loss = bptt_iter_loss.detach()

                    # Update the parameters of the meta optimizer
                    if not (epoch + 1) % bptt_cfg['epochs'] or stop_train:
                        meta_optim.zero_grad()
                        bptt_loss.backward()

                        for name, param in meta_optim.named_parameters():
                            if meta_optim_optim_cfg['grad_clip'] is not None:
                                grad_clip = meta_optim_optim_cfg['grad_clip']
                                param.grad.clamp_(-1.0 * grad_clip, grad_clip)

                            meta_optim_param_grad[name] += param.grad.clone()

                        meta_optim_optim.step()

                        meta_optim.reset(keep_state=True)
                        prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
                        bptt_loss = 0

                    if stop_train:
                        break

                if stop_train:
                    break

            run_train_loss_seqs[seq_name] = torch.tensor(
                run_train_loss_hist).mean()
            with torch.no_grad():
                meta_loss = run_loader(model, meta_loader)
                meta_loss_seqs[seq_name] = meta_loss.item()

        # update meta model on all seqs
        meta_optim.zero_grad()
        meta_optim.load_state_dict(meta_optim_state_dict)
        for name, param in meta_optim.named_parameters():
            param.grad = meta_optim_param_grad[name] / len(db_train.seqs_dict)

            if meta_optim_optim_cfg['grad_clip'] is not None:
                grad_clip = meta_optim_optim_cfg['grad_clip']
                param.grad.clamp_(-1.0 * grad_clip, grad_clip)
        meta_optim_optim.step()

        stop_time = timeit.default_timer()
        if vis_interval is not None and not i % vis_interval:
            meta_metrics = [torch.tensor(list(run_train_loss_seqs.values())).mean(),
                            torch.tensor(list(meta_loss_seqs.values())).min(),
                            torch.tensor(list(meta_loss_seqs.values())).max(),
                            torch.tensor(list(meta_loss_seqs.values())).mean(),
                            epoch + 1,
                            stop_time - start_time]
            vis_dict['meta_metrics_vis'].plot(meta_metrics, i + 1)

        # save_model_to_db(meta_optim, f"update_{i}.model", ex)
        torch.save(meta_optim, os.path.join(save_dir, 'meta', f"meta_run_{i + 1}.model"))
