import copy
import math
import os
import socket
import timeit
import numpy as np
from datetime import datetime
from itertools import count

import networks.vgg_osvos as vo
import sacred
import torch
import torch.multiprocessing as mp
from layers.osvos_layers import class_balanced_cross_entropy_loss, dice_loss
from meta_stopping.meta_optim import MetaOptimizer
from meta_stopping.utils import (compute_loss, dict_to_html,
                                 flat_grads_from_model)
from pytorch_tools.ingredients import (MONGODB_PORT, get_device,
                                       save_model_to_db, set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import LineVis, TextVis
# from sacred.observers import MongoObserver
from util.helper_func import (datasets_and_loaders, early_stopping, grouper,
                              init_parent_model, run_loader, eval_loader, update_dict)

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
        'MEAN seq TRAIN loss',
        'MEAN seq META loss',
        'MEAN seq TEST loss',
        'MEAN seq TEST J',
        'MEAN seq TEST F',
        'RUN TIME']
    opts = dict(
        title=f"OSVOS META - FINAL METRICS (RUN: {_run._id})",
        xlabel='NUM META RUNS',
        ylabel='FINAL METRICS',
        width=750,
        height=300,
        legend=legend)
    vis_dict['meta_metrics_vis'] = LineVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['meta_metrics_vis'].plot([0] * len(legend), 0)

    db_train, *_ = datasets_and_loaders()  # pylint: disable=E1120

    legend = [f"{seq_name}" for seq_name in db_train.seqs_dict.keys()]
    opts = dict(
        title=f"OSVOS META - TEST J (RUN: {_run._id})",
        xlabel='NUM META RUNS',
        ylabel='TEST J',
        width=750,
        height=750,
        legend=legend)
    vis_dict['test_J_seq_vis'] = LineVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['test_J_seq_vis'].plot([0] * len(legend), 0)

    for seq_name in db_train.seqs_dict.keys():
        opts = dict(
            title=f"{seq_name} - MODEL METRICS",
            xlabel='EPOCHS',
            width=450,
            height=300,
            legend=["TRAIN loss", 'META loss', "LR MEAN", "LR STD", "LR MOM MEAN", "WEIGHT DECAY MEAN"])
        vis_dict[f"{seq_name}_model_metrics"] = LineVis(
            opts, env=run_name,  **torch_cfg['vis'])
    return vis_dict


# @ex.capture
def meta_run(i, rank, seq_names, meta_optim_cfg, parent_model_cfg,
             meta_optim_optim_state_dict, meta_optim_state_dict, num_epochs,
             meta_optim_optim_cfg, bptt_cfg, seed, vis_interval, loss_func,
             train_early_stopping, data, return_dict=None, vis_dict=None):
    set_random_seeds(seed)
    device = torch.device(f'cuda:{2 * rank}')
    meta_device = torch.device(f'cuda:{2 * rank + 1}')
    # device = torch.device(f'cuda:{rank}')
    # meta_device = torch.device(f'cuda:{rank}')

    db_train, train_loader, db_test, test_loader, db_meta, meta_loader = datasets_and_loaders(**data)

    model, parent_state_dicts = init_parent_model(parent_model_cfg)
    meta_model, _ = init_parent_model(parent_model_cfg)
    meta_optim = MetaOptimizer(model, meta_model, **meta_optim_cfg)
    meta_optim.load_state_dict(meta_optim_state_dict)
    
    model.to(device)
    meta_model.to(meta_device)
    meta_optim.to(meta_device)

    meta_optim_param_grad = {name: torch.zeros_like(param)
                             for name, param in meta_optim.named_parameters()}

    train_split_X_val_seqs = []
    for file_path in parent_model_cfg['split_val_file_path']:
        seqs = [s.rstrip('\n') for s in open(file_path)]
        train_split_X_val_seqs.append(seqs)

    seqs_metrics = {'meta_loss': {}, 'test_loss': {}, 'test_J': {}, 'test_F': {}, 'train_loss_hist': {}}
    vis_data_seqs = {}

    # make next_meta_frame_ids mutable
    next_meta_frame_ids = return_dict['next_meta_frame_ids'].copy()

    for seq_name in seq_names:
        db_train.set_seq(seq_name)
        db_test.set_seq(seq_name)
        db_meta.set_seq(seq_name)

        vis_data_seqs[seq_name] = []

        bptt_loss = 0
        stop_train = False
        prev_bptt_iter_loss = 0.0
        train_loss_hist = []

        meta_optim_param_grad_seq = {name: torch.zeros_like(param)
                                    for name, param in meta_optim.named_parameters()}

        for seqs_list, p_s_d in zip(train_split_X_val_seqs, parent_state_dicts):
            if seq_name in seqs_list:
                model.load_state_dict(p_s_d)
        model.to(device)
        model.zero_grad()

        meta_optim.load_state_dict(meta_optim_state_dict)
        meta_optim.reset()

        meta_optim_optim = torch.optim.Adam(meta_optim.parameters(),
                                            lr=meta_optim_optim_cfg['lr'])
        # meta_optim_optim.load_state_dict(meta_optim_optim_state_dict)

        # one epoch corresponds to one random transformed first frame of a sequence
        if num_epochs is None:
            epoch_iter = count(start=1)
        else:
            epoch_iter = range(1, num_epochs + 1)

        # meta_loader.dataset.set_random_frame_id()
        if next_meta_frame_ids[seq_name] is not None:
            meta_loader.dataset.frame_id = next_meta_frame_ids[seq_name]
        else:
            meta_loader.dataset.frame_id = data['frame_ids']['meta']

        for epoch in epoch_iter:
            set_random_seeds(seed + epoch + i)
            for train_batch in train_loader:
                train_inputs, train_gts = train_batch['image'], train_batch['gt']
                train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

                train_outputs = model(train_inputs)

                if loss_func == 'cross_entropy':
                    train_loss = class_balanced_cross_entropy_loss(train_outputs[-1], train_gts)
                elif loss_func == 'dice':
                    train_loss = dice_loss(train_outputs[-1], train_gts)
                else:
                    raise NotImplementedError
                train_loss_hist.append(train_loss.item())
                train_loss.backward()

                # meta_optim.seq_id = db_train.get_seq_id()
                meta_optim.train_loss = train_loss.detach()
                meta_model, stop_train = meta_optim.step()
                model.zero_grad()

                stop_train = stop_train or early_stopping(
                    train_loss_hist, **train_early_stopping)

                bptt_iter_loss = 0.0
                for meta_batch in meta_loader:
                    meta_inputs, meta_gts = meta_batch['image'], meta_batch['gt']
                    meta_inputs, meta_gts = meta_inputs.to(
                        meta_device), meta_gts.to(meta_device)
                    
                    meta_outputs = meta_model(meta_inputs)

                    if loss_func == 'cross_entropy':
                        bptt_iter_loss += class_balanced_cross_entropy_loss(meta_outputs[-1], meta_gts)
                    elif loss_func == 'dice':
                        bptt_iter_loss += dice_loss(meta_outputs[-1], meta_gts)
                    else:
                        raise NotImplementedError

                bptt_loss += bptt_iter_loss - prev_bptt_iter_loss
                prev_bptt_iter_loss = bptt_iter_loss.detach()

                # visualization
                lr = meta_optim.state["log_lr"].exp()
                lr_mom = meta_optim.state["lr_mom_logit"].sigmoid()
                weight_decay = meta_optim.state["log_weight_decay"].exp()
                
                vis_data = [train_loss.item(), bptt_loss.item(),
                            lr.mean().item(), lr.std().item(),
                            lr_mom.mean().item(), weight_decay.mean().item()]
                vis_data_seqs[seq_name].append(vis_data)
                
                if vis_interval is not None and vis_dict is not None:
                    vis_dict[f"{seq_name}_model_metrics"].plot(vis_data, epoch)

                # Update the parameters of the meta optimizer
                if not epoch % bptt_cfg['epochs'] or stop_train or epoch == num_epochs:
                    meta_optim.zero_grad()
                    bptt_loss.backward()

                    for name, param in meta_optim.named_parameters():
                        if meta_optim_optim_cfg['grad_clip'] is not None:
                            grad_clip = meta_optim_optim_cfg['grad_clip']
                            param.grad.clamp_(-1.0 * grad_clip, grad_clip)

                        meta_optim_param_grad_seq[name] += param.grad.clone()

                    if meta_optim_optim_cfg['step_in_seq']:
                        meta_optim_optim.step()

                    meta_optim.reset(keep_state=True)
                    prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
                    bptt_loss = 0

                if stop_train:
                    break
            if stop_train:
                break

        with torch.no_grad():
            loss_batches, _ = run_loader(model, meta_loader, loss_func, 'log/meta_results')
            seqs_metrics['meta_loss'][seq_name] = loss_batches.mean()
            
            loss_batches, _, J, F = eval_loader(model, test_loader, loss_func)
            
            next_meta_frame_ids[seq_name] = loss_batches.argmax().item()
            seqs_metrics['test_loss'][seq_name] = loss_batches.mean()
            seqs_metrics['test_J'][seq_name] = J
            seqs_metrics['test_F'][seq_name] = F
        
        seqs_metrics['train_loss_hist'][seq_name] = train_loss_hist

        # normalize over epochs
        for name, grad in meta_optim_param_grad_seq.items():
            meta_optim_param_grad[name] += grad / epoch

    if return_dict is not None:
        meta_optim_param_grad = {name: grad.cpu()
                                 for name, grad in meta_optim_param_grad.items()}

        return_dict['seqs_metrics'] = seqs_metrics
        return_dict['meta_optim_param_grad'] = meta_optim_param_grad
        return_dict['vis_data_seqs'] = vis_data_seqs
        return_dict['next_meta_frame_ids'] = next_meta_frame_ids

    return seqs_metrics, meta_optim_param_grad


@ex.automain
def main(vis_interval, torch_cfg, num_epochs, data, _run, seed, _log, _config,
         save_dir, parent_model_cfg, num_processes, meta_optim_optim_cfg,
         bptt_cfg, train_early_stopping, meta_optim_cfg, loss_func):
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
    meta_optim = MetaOptimizer(model, None)  # pylint: disable=E1120
    meta_optim.init_zero_grad()

    meta_optim_optim = torch.optim.Adam(meta_optim.parameters(),
                                        lr=meta_optim_optim_cfg['lr'])

    db_train, *_ = datasets_and_loaders()  # pylint: disable=E1120

    seqs_metrics = {'meta_loss': {}, 'test_loss': {}, 'test_J': {}, 'test_F': {}}
    next_meta_frame_ids = {n: None for n in db_train.seqs_dict.keys()}
    meta_optim_param_grad = {}

    for i in count():
        start_time = timeit.default_timer()
        meta_optim_state_dict = copy.deepcopy(meta_optim.state_dict())
        meta_optim_optim_state_dict = copy.deepcopy(meta_optim_optim.state_dict())

        meta_optim_param_grad = {name: torch.zeros_like(param)
                                 for name, param in meta_optim.named_parameters()}

        seqs_per_process = math.ceil(len(db_train.seqs_dict) / num_processes)
        for rank, (p, seq_names) in enumerate(zip(processes, grouper(seqs_per_process, db_train.seqs_dict.keys()))):
            seq_names = [n for n in seq_names if n is not None]
            p['return_dict'] = process_manager.dict()
            p['return_dict']['next_meta_frame_ids'] = {n: next_meta_frame_ids[n] for n in seq_names}
            process_args = [i,  rank, seq_names, meta_optim_cfg, parent_model_cfg,
                            meta_optim_optim_state_dict, meta_optim_state_dict,
                            num_epochs, meta_optim_optim_cfg, bptt_cfg, seed,
                            vis_interval, loss_func, train_early_stopping, data,
                            p['return_dict']]
            p['process'] = mp.Process(target=meta_run, args=process_args)
            p['process'].start()

        for p in processes:
            p['process'].join()
            update_dict(seqs_metrics, p['return_dict']['seqs_metrics'])
            
            next_meta_frame_ids.update(p['return_dict']['next_meta_frame_ids'])
            for name in meta_optim_param_grad.keys():
                meta_optim_param_grad[name] += p['return_dict']['meta_optim_param_grad'][name]

            if vis_interval is not None:
                for seq_name, vis_data in p['return_dict']['vis_data_seqs'].items():
                    assert not np.isnan(np.array(vis_data)).any() and np.isfinite(np.array(vis_data)).all()
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
            final_train_loss = torch.tensor([hist[-1] for hist in seqs_metrics['train_loss_hist'].values()])
            meta_metrics = [final_train_loss.mean(),
                            torch.tensor(list(seqs_metrics['meta_loss'].values())).mean(),
                            torch.tensor(list(seqs_metrics['test_loss'].values())).mean(),
                            torch.tensor(list(seqs_metrics['test_J'].values())).mean(),
                            torch.tensor(list(seqs_metrics['test_F'].values())).mean(),
                            timeit.default_timer() - start_time]
            vis_dict['meta_metrics_vis'].plot(meta_metrics, i + 1)

            test_J_seq = [seqs_metrics['test_J'][seq_name] for seq_name in db_train.seqs_dict.keys()]
            vis_dict['test_J_seq_vis'].plot(test_J_seq, i + 1)

        # save_model_to_db(meta_optim, f"update_{i}.model", ex)
        torch.save(meta_optim, os.path.join(save_dir, 'meta', f"meta_run_{i + 1}.model"))
