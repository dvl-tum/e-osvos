import copy
import math
import os
import socket
import timeit
from datetime import datetime
from itertools import count

import networks.vgg_osvos as vo
import sacred
import torch
import torch.multiprocessing as mp
from layers.osvos_layers import class_balanced_cross_entropy_loss
from meta_stopping.meta_optim import MetaOptimizer, SGDFixed
from meta_stopping.utils import (compute_loss, dict_to_html,
                                 flat_grads_from_model)
from pytorch_tools.ingredients import (MONGODB_PORT, get_device,
                                       save_model_to_db, set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import LineVis, TextVis
# from sacred.observers import MongoObserver
from util.helper_func import (datasets_and_loaders, early_stopping, grouper,
                              init_parent_model, run_loader)

torch.multiprocessing.set_start_method("spawn", force=True)

torch_ingredient.add_config('cfgs/torch.yaml')

ex = sacred.Experiment('osvos-meta', ingredients=[torch_ingredient])
ex.add_config('cfgs/meta.yaml')

# if MONGODB_PORT is not None:
#     ex.observers.append(MongoObserver.create(db_name='osvos-meta',
#                                              port=MONGODB_PORT))

MetaOptimizer = ex.capture(MetaOptimizer, prefix='meta_optim_cfg')
datasets_and_loaders = ex.capture(datasets_and_loaders, prefix='data')
early_stopping = ex.capture(early_stopping, prefix='train_early_stopping')


@ex.capture
def init_vis(env_suffix, _config, _run, torch_cfg):
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
        'RUN TIME']
    opts = dict(
        title=f"OSVOS META  (RUN: {_run._id})",
        xlabel='NUM META RUNS',
        width=750,
        height=300,
        legend=legend)
    vis_dict['meta_metrics_vis'] = LineVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['meta_metrics_vis'].plot([0] * len(legend), 0)

    db_train, _, _, _ = datasets_and_loaders()  # pylint: disable=E1120
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


# @ex.capture
def meta_run(i, rank, seq_names, meta_optim_cfg, parent_model_cfg, meta_optim_optim_state_dict,
             meta_optim_state_dict, num_epochs, meta_optim_optim_cfg, bptt_cfg,
             seed, vis_interval, train_early_stopping, data, return_dict=None,
             vis_dict=None):
    set_random_seeds(seed)
    # device = torch.device(f'cuda:{2 * rank}')
    # meta_device = torch.device(f'cuda:{2 * rank + 1}')
    device = torch.device(f'cuda:{rank}')
    meta_device = torch.device(f'cuda:{rank}')

    db_train, train_loader, db_meta, meta_loader = datasets_and_loaders(**data)

    model, parent_state_dicts = init_parent_model(parent_model_cfg)
    meta_model, _ = init_parent_model(parent_model_cfg)
    meta_optim = MetaOptimizer(model, meta_model, **meta_optim_cfg)
    meta_optim.load_state_dict(meta_optim_state_dict)

    model.to(device)
    meta_model.to(meta_device)
    meta_optim.to(meta_device)

    meta_optim_param_grad = {}
    for name, param in meta_optim.named_parameters():
        meta_optim_param_grad[name] = torch.zeros_like(param)

    train_split_X_val_seqs = []
    for i, p in enumerate(parent_model_cfg['split_model_path']):
        if 'split' in p:
            split_file = os.path.join(f'data/DAVIS-2016/train_split_{i + 1}_val.txt')
        else:
            split_file = os.path.join('data/DAVIS-2016/test_seqs.txt')
        seqs = [s.rstrip('\n') for s in open(split_file)]
        train_split_X_val_seqs.append(seqs)

    # one epoch corresponds to one random transformed first frame of a sequence
    if num_epochs is None:
        epoch_iter = count()
        # epoch_iter = range(bptt_cfg['epochs'] * (1 + i // bptt_cfg['runs_per_epoch_extension']))
    else:
        epoch_iter = range(num_epochs)

    run_train_loss_seqs = {}
    meta_loss_seqs = {}
    vis_data_seqs = {}

    for seq_name in seq_names:
        db_train.set_seq(seq_name)
        db_meta.set_seq(seq_name)

        vis_data_seqs[seq_name] = []

        bptt_loss = 0
        stop_train = False
        prev_bptt_iter_loss = 0.0
        run_train_loss_hist = []

        for seqs_list, p_s_d in zip(train_split_X_val_seqs, parent_state_dicts):
            if seq_name in seqs_list:
                model.load_state_dict(p_s_d)
        model.to(device)
        model.zero_grad()

        meta_optim.load_state_dict(meta_optim_state_dict)
        meta_optim.reset()

        meta_optim_optim = torch.optim.Adam(meta_optim.parameters(),
                                            lr=meta_optim_optim_cfg['lr'])
        meta_optim_optim.load_state_dict(meta_optim_optim_state_dict)

        for epoch in epoch_iter:
            set_random_seeds(seed + epoch)
            for train_batch in train_loader:
                train_inputs, train_gts = train_batch['image'], train_batch['gt']
                train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

                train_outputs = model(train_inputs)

                train_loss = class_balanced_cross_entropy_loss(
                    train_outputs[-1], train_gts, size_average=False)
                run_train_loss_hist.append(train_loss.item())
                train_loss.backward()

                meta_optim.seq_id = db_train.get_seq_id()
                meta_model, stop_train = meta_optim.step()
                model.zero_grad()

                stop_train = stop_train or early_stopping(
                    run_train_loss_hist, **train_early_stopping)

                bptt_iter_loss = 0.0
                for meta_batch in meta_loader:
                    meta_inputs, meta_gts = meta_batch['image'], meta_batch['gt']
                    meta_inputs, meta_gts = meta_inputs.to(
                        meta_device), meta_gts.to(meta_device)

                    meta_outputs = meta_model(meta_inputs)

                    bptt_iter_loss += class_balanced_cross_entropy_loss(
                        meta_outputs[-1], meta_gts, size_average=False)

                # visualization
                lr = meta_optim.state["log_lr"].exp().mean()
                lr_mom = meta_optim.state["lr_mom_logit"].sigmoid().mean()
                vis_data = [train_loss.item(), bptt_iter_loss.item(), lr.item(), lr_mom.item()]

                if vis_interval is not None and vis_dict is not None:
                    vis_dict[f"{seq_name}_model_metrics"].plot(vis_data, epoch + 1)

                vis_data_seqs[seq_name].append(vis_data)

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

                    if meta_optim_optim_cfg['step_in_seq']:
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
            meta_loss_seqs[seq_name] = run_loader(model, meta_loader)

    if return_dict is not None:
        meta_optim_param_grad = {name: param.cpu()
                                 for name, param in meta_optim_param_grad.items()}

        return_dict['run_train_loss_seqs'] = run_train_loss_seqs
        return_dict['meta_loss_seqs'] = meta_loss_seqs
        return_dict['meta_optim_param_grad'] = meta_optim_param_grad
        return_dict['vis_data_seqs'] = vis_data_seqs

    return run_train_loss_seqs, meta_loss_seqs, meta_optim_param_grad


@ex.automain
def main(vis_interval, torch_cfg, num_epochs, data, _run, seed, _log, _config,
         save_dir, parent_model_cfg, num_processes, meta_optim_optim_cfg,
         bptt_cfg, train_early_stopping, meta_optim_cfg):
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))
    set_random_seeds(seed)

    if vis_interval is not None:
        vis_dict = init_vis()  # pylint: disable=E1120

    #
    # Meta model
    #
    process_manager = mp.Manager()
    processes = [dict() for _ in range(num_processes)]

    model, _ = init_parent_model(parent_model_cfg)
    meta_optim = MetaOptimizer(model, model)  # pylint: disable=E1120

    ### init zero gradients
    output = meta_optim(torch.ones(1, 8))
    (output[0] + output[1]).sum().backward()
    meta_optim.zero_grad()
    ### init zero gradients

    meta_optim_optim = torch.optim.Adam(meta_optim.parameters(),
                                        lr=meta_optim_optim_cfg['lr'])

    db_train, _, _, _ = datasets_and_loaders()  # pylint: disable=E1120

    meta_loss_seqs = {}
    run_train_loss_seqs = {}
    meta_optim_param_grad = {}
    for i in count():
        start_time = timeit.default_timer()
        meta_optim_state_dict = copy.deepcopy(meta_optim.state_dict())
        meta_optim_optim_state_dict = copy.deepcopy(meta_optim_optim.state_dict())

        for name, param in meta_optim.named_parameters():
            meta_optim_param_grad[name] = torch.zeros_like(param).cpu()

        seqs_per_process = math.ceil(len(db_train.seqs_dict) / num_processes)
        for rank, (p, seq_names) in enumerate(zip(processes, grouper(seqs_per_process, db_train.seqs_dict.keys()))):
            seq_names = [n for n in seq_names if n is not None]
            p['return_dict'] = process_manager.dict()
            process_args = [i,  rank, seq_names, meta_optim_cfg, parent_model_cfg,
                            meta_optim_optim_state_dict, meta_optim_state_dict,
                            num_epochs, meta_optim_optim_cfg, bptt_cfg, seed,
                            vis_interval, train_early_stopping, data, p['return_dict']]
            p['process'] = mp.Process(target=meta_run, args=process_args)
            p['process'].start()

        for p in processes:
            p['process'].join()
            run_train_loss_seqs.update(p['return_dict']['run_train_loss_seqs'])
            meta_loss_seqs.update(p['return_dict']['meta_loss_seqs'])

            for name in meta_optim_param_grad.keys():
                meta_optim_param_grad[name] += p['return_dict']['meta_optim_param_grad'][name]

            if vis_interval is not None:
                for seq_name, vis_data in p['return_dict']['vis_data_seqs'].items():
                    vis_dict[f"{seq_name}_model_metrics"].reset()
                    for epoch, vis_datum in enumerate(vis_data):
                        vis_dict[f"{seq_name}_model_metrics"].plot(vis_datum, epoch + 1)

        meta_optim.zero_grad()
        for name, param in meta_optim.named_parameters():
            param.grad = meta_optim_param_grad[name].to(
                param.grad.device) / len(db_train.seqs_dict)

            if meta_optim_optim_cfg['grad_clip'] is not None:
                grad_clip = meta_optim_optim_cfg['grad_clip']
                param.grad.clamp_(-1.0 * grad_clip, grad_clip)

        meta_optim_optim.step()

        if vis_interval is not None and not i % vis_interval:
            meta_metrics = [torch.tensor(list(run_train_loss_seqs.values())).mean(),
                            torch.tensor(list(meta_loss_seqs.values())).min(),
                            torch.tensor(list(meta_loss_seqs.values())).max(),
                            torch.tensor(list(meta_loss_seqs.values())).mean(),
                            timeit.default_timer() - start_time]
            vis_dict['meta_metrics_vis'].plot(meta_metrics, i + 1)

        # save_model_to_db(meta_optim, f"update_{i}.model", ex)
        # torch.save(meta_optim, os.path.join(save_dir, 'meta', f"meta_run_{i + 1}.model"))
