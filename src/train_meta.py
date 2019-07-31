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
from util.helper_func import (datasets_and_loaders, early_stopping, grouper, train_val,
                              init_parent_model, run_loader, eval_loader, update_dict)

torch.multiprocessing.set_start_method("spawn", force=True)
torch_ingredient.add_config('cfgs/torch.yaml')

ex = sacred.Experiment('osvos-meta', ingredients=[torch_ingredient])
ex.add_config('cfgs/meta.yaml')

# if MONGODB_PORT is not None:
#     ex.observers.append(MongoObserver.create(db_name='osvos-meta',
#                                              port=MONGODB_PORT))

MetaOptimizer = ex.capture(MetaOptimizer, prefix='meta_optim_cfg')
datasets_and_loaders = ex.capture(datasets_and_loaders, prefix='data_cfg')
early_stopping = ex.capture(early_stopping, prefix='train_early_stopping')
init_parent_model = ex.capture(init_parent_model)
train_val = ex.capture(train_val)


@ex.capture
def init_vis(env_suffix, _config, _run, val_seq_name, torch_cfg):
    run_name = f"{_run.experiment_info['name']}_{env_suffix}"
    vis_dict = {}

    opts = dict(title=f"CONFIG and NON META BASELINE (RUN: {_run._id})",
                width=300, height=1250)
    vis_dict['config_vis'] = TextVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['config_vis'].plot(dict_to_html(_config))

    legend = [
        'MEAN seq TRAIN loss',
        'MEAN seq META loss',
        'MEAN seq loss',
        'MEAN seq J',
        'MEAN seq F',
        'RUN TIME [min]']
    opts = dict(
        title=f"FINAL TRAIN METRICS (RUN: {_run._id})",
        xlabel='NUM META RUNS',
        ylabel='FINAL METRICS',
        width=750,
        height=300,
        legend=legend)
    vis_dict['meta_metrics_vis'] = LineVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['meta_metrics_vis'].plot([0] * len(legend), 0)

    if val_seq_name is not None:
        db_train, *_ = datasets_and_loaders(seq_name=val_seq_name)  # pylint: disable=E1120
        legend = ['INIT MEAN', 'MEAN'] + [f"{seq_name}" for seq_name in db_train.seqs_dict.keys()]
        opts = dict(
            title=f"VAL J (RUN: {_run._id})",
            xlabel='NUM META RUNS',
            ylabel='VAL J',
            width=750,
            height=750,
            legend=legend)
        vis_dict['val_J_seq_vis'] = LineVis(opts, env=run_name, **torch_cfg['vis'])
        vis_dict['val_J_seq_vis'].plot([0] * len(legend), 0)

    db_train, *_ = datasets_and_loaders()  # pylint: disable=E1120

    legend = ['TRAIN loss', 'META loss', 'LR MEAN', 'LR STD',
              'LR MOM MEAN', 'WEIGHT DECAY MEAN']
    for seq_name in db_train.seqs_dict.keys():
        opts = dict(
            title=f"SEQ METRICS - {seq_name}",
            xlabel='EPOCHS',
            width=450,
            height=300,
            legend=legend)
        vis_dict[f"{seq_name}_model_metrics"] = LineVis(
            opts, env=run_name,  **torch_cfg['vis'])
    return vis_dict


def meta_run(i, rank, seq_names, meta_optim_optim_state_dict, meta_optim_state_dict, 
             meta_optim_param_grad, next_meta_frame_ids, _config, return_dict):
    loss_func = _config['loss_func']
    num_epochs = _config['num_epochs']

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    set_random_seeds(_config['seed'])
    
    device = torch.device(f'cuda:{(2 * rank) + 1}')
    meta_device = torch.device(f'cuda:{(2 * rank + 1) + 1}')

    db_train, train_loader, _, _, db_meta, meta_loader = datasets_and_loaders(**_config['data_cfg'])

    model, _ = init_parent_model(_config['parent_model_path'])
    meta_model, _ = init_parent_model(_config['parent_model_path'])

    meta_optim = MetaOptimizer(model, meta_model, **_config['meta_optim_cfg'])
    meta_optim.load_state_dict(meta_optim_state_dict)

    model.to(device)
    meta_model.to(meta_device)
    meta_optim.to(meta_device)

    seqs_metrics = {'train_loss': {}, 'meta_loss': {}, 'loss': {}, 'J': {}, 'F': {}}
    vis_data_seqs = {}

    for seq_name in seq_names:
        db_train.set_seq(seq_name)
        # db_test.set_seq(seq_name)
        db_meta.set_seq(seq_name)

        vis_data_seqs[seq_name] = []

        bptt_loss = 0
        stop_train = False
        prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
        train_loss_hist = []

        meta_optim_param_grad_seq = {name: torch.zeros_like(param)
                                     for name, param in meta_optim.named_parameters()}

        meta_optim.load_state_dict(meta_optim_state_dict)
        meta_optim.reset()

        meta_optim_optim = torch.optim.Adam(meta_optim.parameters(),
                                            lr=_config['meta_optim_optim_cfg']['lr'])
        # meta_optim_optim.load_state_dict(meta_optim_optim_state_dict)

        # one epoch corresponds to one random transformed first frame of a sequence
        if num_epochs is None:
            epoch_iter = count(start=1)
        else:
            epoch_iter = range(1, num_epochs + 1)

        if next_meta_frame_ids[seq_name] is not None:
            meta_loader.dataset.frame_id = next_meta_frame_ids[seq_name]
        else:
            meta_loader.dataset.frame_id = _config['data_cfg']['frame_ids']['meta']

        for epoch in epoch_iter:
            set_random_seeds(_config['seed'] + epoch)
            
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

                meta_optim.train_loss = train_loss.detach()
                meta_model, stop_train = meta_optim.step()
                model.zero_grad()

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

                stop_train = stop_train or early_stopping(
                    train_loss_hist, **_config['train_early_stopping_cfg'])

                # update params of meta optim
                if not epoch % _config['bptt_epochs'] or stop_train or epoch == num_epochs:
                    meta_optim.zero_grad()
                    bptt_loss.backward()

                    for name, param in meta_optim.named_parameters():
                        grad_clip = _config['meta_optim_optim_cfg']['grad_clip']
                        if grad_clip is not None:
                            param.grad.clamp_(-1.0 * grad_clip, grad_clip)

                        meta_optim_param_grad_seq[name] += param.grad.clone()

                    if _config['meta_optim_optim_cfg']['step_in_seq']:
                        meta_optim_optim.step()

                    meta_optim.reset(keep_state=True)
                    prev_bptt_iter_loss.zero_().detach_()
                    bptt_loss = 0

                if stop_train:
                    break
            if stop_train:
                break

        loss_batches, _ = run_loader(model, meta_loader, loss_func, 'log/meta_results')
        seqs_metrics['meta_loss'][seq_name] = loss_batches.mean()

        # loss_batches, _, J, F = eval_loader(model, test_loader, loss_func)
        # next_meta_frame_ids[seq_name] = loss_batches.argmax().item()

        # seqs_metrics['loss'][seq_name] = loss_batches.mean()
        # seqs_metrics['J'][seq_name] = J
        # seqs_metrics['F'][seq_name] = F
        seqs_metrics['train_loss'][seq_name] = train_loss_hist[-1]

        # normalize over epochs
        for name, grad in meta_optim_param_grad_seq.items():
            meta_optim_param_grad[name] += grad.cpu() / epoch

    return_dict['seqs_metrics'] = seqs_metrics
    return_dict['vis_data_seqs'] = vis_data_seqs


def validate(meta_optim_state_dict, _config, return_dict):
    seed = _config['seed']
    loss_func = _config['loss_func']
    
    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_random_seeds(seed)
    device = torch.device(f'cuda:0')

    model, _ = init_parent_model(_config['parent_model_path'])
    meta_model, _ = init_parent_model(_config['parent_model_path'])

    meta_optim = MetaOptimizer(model, meta_model, **_config['meta_optim_cfg'])
    meta_optim.load_state_dict(meta_optim_state_dict)

    model.to(device)
    meta_model.to(device)
    meta_optim.to(device)

    data_cfg = copy.deepcopy(_config['data_cfg'])
    data_cfg['seq_name'] = _config['val_seq_name']
    db_train, train_loader, db_test, test_loader, *_ = datasets_and_loaders(**data_cfg)  # pylint: disable=E1120

    def early_stopping_func(loss_hist): return early_stopping(loss_hist, **_config['train_early_stopping_cfg'])

    init_test_J_seq = []
    test_J_seq = []
    for seq_name in db_train.seqs_dict.keys():
        db_train.set_seq(seq_name)
        db_test.set_seq(seq_name)

        meta_optim.reset()
        meta_optim.eval()

        model.zero_grad()
        _, _, J, _ = eval_loader(model, test_loader, loss_func)
        init_test_J_seq.append(J)

        train_val(  # pylint: disable=E1120
            model, train_loader, None, meta_optim, _config['num_epochs'], seed,
            early_stopping_func=early_stopping_func,
            validate_inter=None,
            loss_func=loss_func)

        _, _, J, _ = eval_loader(model, test_loader, loss_func)
        test_J_seq.append(J)

    return_dict['init_test_J_seq'] = init_test_J_seq
    return_dict['test_J_seq'] = test_J_seq


@ex.automain
def main(vis_interval, save_dir, num_processes, meta_optim_optim_cfg, seed, _config):
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))
    set_random_seeds(seed)

    if vis_interval is not None:
        vis_dict = init_vis()  # pylint: disable=E1120

    db_train, *_ = datasets_and_loaders()  # pylint: disable=E1120

    #
    # Meta model
    #
    seqs_per_process = math.ceil(len(db_train.seqs_dict) / num_processes)
    process_manager = mp.Manager()
    meta_processes = [dict() for _ in range(num_processes)]
    val_process = {}

    model, parent_state_dict = init_parent_model()
    meta_model, _ = init_parent_model()

    model.load_state_dict(parent_state_dict)
    meta_model.load_state_dict(parent_state_dict)

    meta_optim = MetaOptimizer(model, meta_model)  # pylint: disable=E1120
    meta_optim.init_zero_grad()

    del model
    del meta_model
    # meta_optim.to(device)
    # meta_model.to(device)
    # model.to(device)

    meta_optim_params = [{'params': [p for n, p in meta_optim.named_parameters() if 'model_init' in n], 'lr': meta_optim_optim_cfg['model_init_lr']},
                         {'params': [p for n, p in meta_optim.named_parameters() if 'log_init_lr' in n], 'lr': meta_optim_optim_cfg['log_init_lr_lr']},
                         {'params': [p for n, p in meta_optim.named_parameters() if 'model_init' not in n and 'log_init_lr' not in n], 'lr': meta_optim_optim_cfg['lr']}]
    meta_optim_optim = torch.optim.Adam(meta_optim_params,
                                        lr=meta_optim_optim_cfg['lr'])

    seqs_metrics = {'train_loss': {}, 'meta_loss': {}, 'loss': {}, 'J': {}, 'F': {}}
    next_meta_frame_ids = process_manager.dict({n: None for n in db_train.seqs_dict.keys()})
    meta_optim_param_grad = process_manager.dict({name: torch.zeros_like(param).cpu()
                                                  for name, param in meta_optim.named_parameters()})

    for i in count(start=1):
        start_time = timeit.default_timer()

        # start validation
        if vis_interval is not None and not val_process:
            val_process['num_meta_run'] = i
            val_process['return_dict'] = process_manager.dict()
            process_args = [meta_optim.state_dict(), _config, val_process['return_dict']]
            val_process['process'] = mp.Process(target=validate, args=process_args)
            val_process['process'].start()

        # set meta optim gradients to zero
        for p in meta_optim_param_grad.values():
            p.zero_()

        # set random next frame ids
        for seq_name in db_train.seqs_dict.keys():
            db_train.set_seq(seq_name)
            db_train.set_random_frame_id()
            next_meta_frame_ids[seq_name] = db_train.frame_id
        
        # start processes 
        for rank, (p, seq_names) in enumerate(zip(meta_processes, grouper(seqs_per_process, db_train.seqs_dict.keys()))):
            # filter None values from grouper
            seq_names = [n for n in seq_names if n is not None]

            p['return_dict'] = process_manager.dict()

            process_args = [i, rank, seq_names, meta_optim_optim.state_dict(),
                            meta_optim.state_dict(), meta_optim_param_grad,
                            next_meta_frame_ids, _config, p['return_dict']]
            p['process'] = mp.Process(target=meta_run, args=process_args)
            p['process'].start()

        # join processes
        for p in meta_processes:
            p['process'].join()
            update_dict(seqs_metrics, p['return_dict']['seqs_metrics'])

            if vis_interval is not None:
                for seq_name, vis_data in p['return_dict']['vis_data_seqs'].items():
                    # Visdom throws no error for infinte or NaN values
                    assert not np.isnan(np.array(vis_data)).any() and np.isfinite(np.array(vis_data)).all()

                    vis_dict[f"{seq_name}_model_metrics"].reset()
                    for epoch, vis_datum in enumerate(vis_data):
                        vis_dict[f"{seq_name}_model_metrics"].plot(vis_datum, epoch + 1)

        # visualize meta runs
        if vis_interval is not None and (i == 1 or not i % vis_interval):
            meta_metrics = [torch.tensor(list(m.values())).mean() for m in seqs_metrics.values()]
            meta_metrics.append((timeit.default_timer() - start_time) / 60)                

            vis_dict['meta_metrics_vis'].plot(meta_metrics, i)

        # optimize meta_optim
        meta_optim.zero_grad()
        for name, param in meta_optim.named_parameters():
            param.grad = meta_optim_param_grad[name] / len(db_train.seqs_dict)

            if meta_optim_optim_cfg['grad_clip'] is not None:
                grad_clip = meta_optim_optim_cfg['grad_clip']
                param.grad.clamp_(-1.0 * grad_clip, grad_clip)

        meta_optim_optim.step()

        # join and visualize validation
        if val_process and not val_process['process'].is_alive():
            val_process['process'].join()

            test_J_seq = val_process['return_dict']['test_J_seq']
            test_J_seq.insert(0, torch.tensor(test_J_seq).mean())
            test_J_seq.insert(0, torch.tensor(
                val_process['return_dict']['init_test_J_seq']).mean())
            vis_dict['val_J_seq_vis'].plot(test_J_seq, val_process['num_meta_run'])

            val_process = {}

        # save_model_to_db(meta_optim, f"update_{i}.model", ex)
        torch.save(meta_optim, os.path.join(save_dir, 'meta', f"meta_run_{i + 1}.model"))
