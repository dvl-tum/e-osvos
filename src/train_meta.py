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
from pytorch_tools.ingredients import (save_model_to_db, set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import LineVis, TextVis
from davis import cfg as davis_cfg
from util.helper_func import (datasets_and_loaders, early_stopping, grouper, train_val,
                              init_parent_model, run_loader, eval_loader, update_dict)

torch_ingredient.add_config('cfgs/torch.yaml')

ex = sacred.Experiment('osvos-meta', ingredients=[torch_ingredient])
ex.add_config('cfgs/meta.yaml')


MetaOptimizer = ex.capture(MetaOptimizer, prefix='meta_optim_cfg')
datasets_and_loaders = ex.capture(datasets_and_loaders, prefix='data_cfg')
early_stopping = ex.capture(early_stopping, prefix='train_early_stopping')
init_parent_model = ex.capture(init_parent_model)
train_val = ex.capture(train_val)


@ex.capture
def init_vis(env_suffix: str, _config: dict, _run: sacred.run.Run,
             torch_cfg: dict, datasets: dict):
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

    for dataset_name, dataset in datasets.items():
        db, *_ = datasets_and_loaders(dataset)  # pylint: disable=E1120
        legend = ['INIT MEAN', 'MEAN'] + [f"{seq_name}" for seq_name in db.seqs_dict.keys()]
        opts = dict(
            title=f"{dataset_name.upper()} J (RUN: {_run._id})",
            xlabel='NUM META RUNS',
            ylabel=f'{dataset_name.upper()} J',
            width=750,
            height=750,
            legend=legend)
        vis_dict[f'{dataset_name}_J_seq_vis'] = LineVis(opts, env=run_name, **torch_cfg['vis'])

    db_train, *_ = datasets_and_loaders(datasets['train'])  # pylint: disable=E1120
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

    model, _ = init_parent_model()
    legend = ['MEAN', 'STD'] + [f"{n}" for n, p in model.named_parameters()
                         if p.requires_grad]
    opts = dict(
        title=f"INIT LR (RUN: {_run._id})",
        xlabel='NUM META RUNS',
        ylabel='LR',
        width=750,
        height=750,
        legend=legend)
    vis_dict['init_lr_vis'] = LineVis(opts, env=run_name, **torch_cfg['vis'])

    return vis_dict


def setup_davis_eval(data_cfg: dict):
    davis_cfg.YEAR = 2016
    davis_cfg.PATH.ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    davis_cfg.PATH.DATA = os.path.abspath(os.path.join(davis_cfg.PATH.ROOT, data_cfg['root_dir']))
    davis_cfg.PATH.SEQUENCES = os.path.join(davis_cfg.PATH.DATA, "JPEGImages", davis_cfg.RESOLUTION)
    davis_cfg.PATH.ANNOTATIONS = os.path.join(davis_cfg.PATH.DATA, "Annotations", davis_cfg.RESOLUTION)
    davis_cfg.PATH.PALETTE = os.path.abspath(os.path.join(davis_cfg.PATH.ROOT, 'data/palette.txt'))


def meta_run(i: int, rank: int, seq_names: list, meta_optim_state_dict: dict,
             meta_optim_param_grad: dict, random_frame_rng_state: torch.ByteTensor,
             _config: dict, dataset: str, return_dict: dict):
    loss_func = _config['loss_func']
    num_epochs = _config['num_epochs']

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_random_seeds(_config['seed'])

    setup_davis_eval(_config['data_cfg'])

    device = torch.device(f'cuda:{(2 * rank) + 2}')
    meta_device = torch.device(f'cuda:{(2 * rank + 1) + 2}')

    db_train, train_loader, db_test, test_loader, db_meta, meta_loader = datasets_and_loaders(dataset, **_config['data_cfg'])

    # data_cfg = copy.deepcopy(_config['data_cfg'])
    # data_cfg['batch_sizes']['train'] = 1
    # data_cfg['batch_sizes']['meta'] = 1
    # data_cfg['random_train_transform'] = False
    # db_train_matching_info, train_loader_matching_info, _, _, db_meta_matching_info, meta_loader_matching_info = datasets_and_loaders(dataset, **data_cfg)  # pylint: disable=E1120

    model, parent_state_dict = init_parent_model(_config['parent_model_path'])
    meta_model, _ = init_parent_model(_config['parent_model_path'])

    meta_optim = MetaOptimizer(model, meta_model, **_config['meta_optim_cfg'])
    meta_optim.load_state_dict(meta_optim_state_dict)

    model.to(device)
    meta_model.to(meta_device)
    meta_optim.to(meta_device)

    seqs_metrics = ['train_loss', 'meta_loss', 'loss', 'J', 'F']
    seqs_metrics = {m: {n: [] for n in seq_names} for m in seqs_metrics}

    vis_data_seqs = {}

    for _ in range(_config['num_seq_epochs_per_step']):
        for seq_name in seq_names:
            db_train.set_seq(seq_name)
            db_test.set_seq(seq_name)
            db_meta.set_seq(seq_name)

            vis_data_seqs[seq_name] = []

            bptt_loss = 0
            stop_train = False
            prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
            train_loss_hist = []

            meta_optim_param_grad_seq = {name: torch.zeros_like(param)
                                        for name, param in meta_optim.named_parameters()}

            model.load_state_dict(parent_state_dict)
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

            torch.set_rng_state(random_frame_rng_state)
            db_train.set_random_frame_id()
            db_meta.set_random_frame_id()
            random_frame_rng_state = torch.get_rng_state()

            # # first frame matching
            # db_train_matching_info.set_seq(seq_name)
            # db_meta_matching_info.set_seq(seq_name)
            # db_train_matching_info.frame_id = db_train.frame_id
            # db_meta_matching_info.frame_id = db_meta.frame_id
            # model.eval()
            # with torch.no_grad():
            #     for train_batch in train_loader_matching_info:
            #         train_inputs, train_gts = train_batch['image'], train_batch['gt']
            #         train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

            #         train_mean_img = train_inputs.mean().expand_as(train_inputs)
            #         foreground_train_inputs = train_mean_img * (1 - train_gts) + train_inputs * train_gts
            #         train_encoder_feat = model.encoder(foreground_train_inputs)[0] # c5

            #     for meta_batch in meta_loader_matching_info:
            #         meta_inputs, meta_gts = meta_batch['image'], meta_batch['gt']
            #         meta_inputs, meta_gts = meta_inputs.to(device), meta_gts.to(device)

            #         meta_outputs = model(meta_inputs)[0]
            #         meta_preds = torch.sigmoid(meta_outputs)
            #         meta_preds = meta_preds.ge(0.5).float()

            #         meta_mean_img = meta_inputs.mean().expand_as(meta_inputs)
            #         foreground_meta_inputs = meta_mean_img * (1 - meta_preds) + meta_inputs * meta_preds
            #         meta_encoder_feat = model.encoder(foreground_meta_inputs)[0] # c5

            #         train_encoder_feat = train_encoder_feat.view(512, -1).mean(dim=1)
            #         meta_encoder_feat = meta_encoder_feat.view(512, -1).mean(dim=1)
            #         matching_info = train_encoder_feat.sub(meta_encoder_feat).abs().mean() * 100
            
            # meta_optim.matching_info = matching_info.detach()
            # print('meta run', seq_name, matching_info)

            # for name, m in model.named_modules():
            #     if isinstance(m, torch.nn.BatchNorm2d):
            #         print(m.training, m.track_running_stats, m.running_mean.abs().mean())
            #         break

            for epoch in epoch_iter:
                set_random_seeds(_config['seed'] + epoch + i)

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

                    # for name, m in model.named_modules():
                    #     if isinstance(m, torch.nn.BatchNorm2d):
                    #         print(m.training, m.track_running_stats, m.running_mean.abs().mean())
                    #         break
 
                    if epoch == 1:
                        meta_optim.train_loss = torch.zeros_like(train_loss)
                    else:
                        meta_optim.train_loss = train_loss.detach() - meta_optim.prev_train_loss
                    meta_optim.prev_train_loss = train_loss.detach()                        
                    # print('meta run', seq_name, meta_optim.train_loss.item())
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

            loss_batches, _ = run_loader(model, meta_loader, loss_func)
            seqs_metrics['meta_loss'][seq_name].append(loss_batches.mean())

            # loss_batches, _, J, F = eval_loader(model, test_loader, loss_func)
            # next_meta_frame_ids[seq_name] = loss_batches.argmax().item()

            # seqs_metrics['loss'][seq_name] = loss_batches.mean()
            # seqs_metrics['J'][seq_name] = J
            # seqs_metrics['F'][seq_name] = F
            seqs_metrics['train_loss'][seq_name].append(train_loss_hist[-1])

            # normalize over epochs
            for name, grad in meta_optim_param_grad_seq.items():
                meta_optim_param_grad[name] += grad.cpu() / _config['num_seq_epochs_per_step']

    # compute mean over num_seq_epochs_per_step
    seqs_metrics = {metric_name: {seq_name: torch.tensor(vv).mean() for seq_name, vv in v.items()}
                    for metric_name, v in seqs_metrics.items()}

    return_dict['seqs_metrics'] = seqs_metrics
    return_dict['vis_data_seqs'] = vis_data_seqs
    return_dict['random_frame_rng_state'] = random_frame_rng_state


def evaluate(rank, dataset, meta_optim_state_dict: dict, _config: dict, return_dict: dict):
    seed = _config['seed']
    loss_func = _config['loss_func']

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    set_random_seeds(seed)
    device = torch.device(f'cuda:{rank}')
    
    setup_davis_eval(_config['data_cfg'])
    
    if 'test' in dataset:
        model, parent_state_dict = init_parent_model(_config['val_parent_model_path'])
        meta_model, _ = init_parent_model(_config['val_parent_model_path'])
    else:
        model, parent_state_dict = init_parent_model(_config['parent_model_path'])
        meta_model, _ = init_parent_model(_config['parent_model_path'])

    meta_optim = MetaOptimizer(model, meta_model, **_config['meta_optim_cfg'])
    meta_optim.load_state_dict(meta_optim_state_dict)
    
    model.to(device)
    meta_model.to(device)
    meta_optim.to(device)
    
    db_train, train_loader, db_test, test_loader, *_ = datasets_and_loaders(dataset, **_config['data_cfg'])  # pylint: disable=E1120

    # data_cfg = copy.deepcopy(_config['data_cfg'])
    # data_cfg['batch_sizes']['train'] = 1
    # data_cfg['batch_sizes']['test'] = 1
    # data_cfg['random_train_transform'] = False
    # db_train_matching_info, train_loader_matching_info, db_test_matching_info, test_loader_matching_info, * \
    #     _ = datasets_and_loaders(dataset, **_config['data_cfg'])  # pylint: disable=E1120

    def early_stopping_func(loss_hist):
        return early_stopping(loss_hist, **_config['train_early_stopping_cfg'])
    
    init_J_seq = []
    J_seq = []
    for seq_name in db_train.seqs_dict.keys():
        db_train.set_seq(seq_name)
        db_test.set_seq(seq_name)

        model.load_state_dict(parent_state_dict)
        meta_optim.reset()
        meta_optim.eval()

        # seq temporal info
        # db_train_matching_info.set_seq(seq_name)
        # db_test_matching_info.set_seq(seq_name)
        # model.eval()
        
        # for name, m in model.named_modules():
        #     if isinstance(m, torch.nn.BatchNorm2d):
        #         print(m.training, m.track_running_stats, m.running_mean.abs().mean())
        #         break

        # with torch.no_grad():
        #     for train_batch in train_loader_matching_info:
        #         train_inputs, train_gts = train_batch['image'], train_batch['gt']
        #         train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

        #         train_mean_img = train_inputs.mean().expand_as(train_inputs)
        #         foreground_train_inputs = train_mean_img * (1 - train_gts) + train_inputs * train_gts
        #         train_encoder_feat = model.encoder(foreground_train_inputs)[0] # c5

        #     seq_temporal_infos = []
        #     for test_batch in test_loader_matching_info:
        #         test_inputs, test_gts = test_batch['image'], test_batch['gt']
        #         test_inputs, test_gts = test_inputs.to(device), test_gts.to(device)

        #         test_outputs = model(test_inputs)[0]
        #         test_preds = torch.sigmoid(test_outputs)
        #         test_preds = test_preds.ge(0.5).float()

        #         test_mean_img = test_inputs.mean().expand_as(test_inputs)
        #         foreground_test_inputs = test_mean_img * (1 - test_preds) + test_inputs * test_preds
        #         test_encoder_feat = model.encoder(foreground_test_inputs)[0] # c5

        #         train_encoder_feat = train_encoder_feat.view(512, -1).mean(dim=1)
        #         test_encoder_feat = test_encoder_feat.view(512, -1).mean(dim=1)

        #         seq_temporal_infos.append(train_encoder_feat.sub(test_encoder_feat).abs().mean().item())

        #     meta_optim.matching_info = torch.tensor(seq_temporal_infos).mean() * 100
        # print('validate', seq_name, meta_optim.matching_info)
        model.zero_grad()
        _, _, J, _, preds = eval_loader(model, test_loader, loss_func, return_preds=True)
        init_test_J_seq.append(J)

        _, _, J, _, _ = eval_loader(model, test_loader, loss_func, return_preds=True)
        init_J_seq.append(J)

            seq_temporal_info = train_gts[0:1].sub(preds.float()).abs()
            seq_temporal_info = seq_temporal_info.view(seq_temporal_info.size(0), -1).mean(dim=1)
            seq_temporal_info = seq_temporal_info.max()
            meta_optim.seq_temporal_info = seq_temporal_info * 100

        train_val(  # pylint: disable=E1120
            model, train_loader, None, meta_optim, _config['num_epochs'], seed,
            early_stopping_func=early_stopping_func,
            validate_inter=None,
            loss_func=loss_func)

        _, _, J, _ = eval_loader(model, test_loader, loss_func)
        J_seq.append(J)

    return_dict['init_J_seq'] = init_J_seq
    return_dict['J_seq'] = J_seq


@ex.automain
def main(save_dir: str, num_processes: int, datasets: dict,
         meta_optim_optim_cfg: dict, seed: int, _config: dict):
    mp.set_start_method("spawn")

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))
    set_random_seeds(seed)

    vis_dict = init_vis()  # pylint: disable=E1120

    db_train, *_ = datasets_and_loaders(datasets['train'])  # pylint: disable=E1120

    #
    # processes
    #
    seqs_per_process = math.ceil(len(db_train.seqs_dict) / num_processes)
    process_manager = mp.Manager()
    meta_processes = [dict() for _ in range(num_processes)]
    eval_processes = {n: {} for n in datasets.keys()}
    
    #
    # Meta model
    #
    model, parent_state_dict = init_parent_model()
    meta_model, _ = init_parent_model()

    model.load_state_dict(parent_state_dict)
    meta_model.load_state_dict(parent_state_dict)

    meta_optim = MetaOptimizer(model, meta_model)  # pylint: disable=E1120
    meta_optim.init_zero_grad()

    del model
    del meta_model

    meta_optim_params = [{'params': [p for n, p in meta_optim.named_parameters() if 'model_init' in n],
                          'lr': meta_optim_optim_cfg['model_init_lr']},
                         {'params': [p for n, p in meta_optim.named_parameters() if 'log_init_lr' in n],
                          'lr': meta_optim_optim_cfg['log_init_lr_lr']},
                         {'params': [p for n, p in meta_optim.named_parameters() if 'model_init' not in n and 'log_init_lr' not in n],
                          'lr': meta_optim_optim_cfg['lr']}]
    meta_optim_optim = torch.optim.Adam(meta_optim_params,
                                        lr=meta_optim_optim_cfg['lr'])

    seqs_metrics = {'train_loss': {}, 'meta_loss': {}, 'loss': {}, 'J': {}, 'F': {}}
    meta_optim_param_grad = process_manager.dict({name: torch.zeros_like(param).cpu()
                                                  for name, param in meta_optim.named_parameters()})

    random_frame_rng_state = torch.get_rng_state()

    for i in count(start=1):
        start_time = timeit.default_timer()
        
        # start train and val evaluation
        for rank, (dataset_name, p) in enumerate(eval_processes.items()):
            if 'process' not in p or not p['process'].is_alive():
                p['num_meta_run'] = i
                p['return_dict'] = process_manager.dict()
                process_args = [rank, datasets[dataset_name], meta_optim.state_dict(),
                                _config, p['return_dict']]
                p['process'] = mp.Process(target=evaluate, args=process_args)
                p['process'].start()
        
        # set meta optim gradients to zero
        for p in meta_optim_param_grad.values():
            p.zero_()

        # start meta run processes
        for rank, (p, seq_names) in enumerate(zip(meta_processes, grouper(seqs_per_process, db_train.seqs_dict.keys()))):
            # filter None values from grouper
            seq_names = [n for n in seq_names if n is not None]

            p['return_dict'] = process_manager.dict()

            process_args = [i, rank, seq_names, meta_optim.state_dict(),
                            meta_optim_param_grad, random_frame_rng_state,
                            _config, datasets['train'], p['return_dict']]
            p['process'] = mp.Process(target=meta_run, args=process_args)
            p['process'].start()

        # join meta run processes
        for p in meta_processes:
            p['process'].join()
            update_dict(seqs_metrics, p['return_dict']['seqs_metrics'])
            random_frame_rng_state = p['return_dict']['random_frame_rng_state']

        # visualize meta metrics and seq runs
        meta_metrics = [torch.tensor(list(m.values())).mean()
                        for m in seqs_metrics.values()]
        meta_metrics.append((timeit.default_timer() - start_time) / 60)

        vis_dict['meta_metrics_vis'].plot(meta_metrics, i)

        meta_init_lr = [meta_optim.log_init_lr.exp().mean(),
                        meta_optim.log_init_lr.exp().std()]
        meta_init_lr += meta_optim.log_init_lr.exp().detach().numpy().tolist()
        vis_dict['init_lr_vis'].plot(meta_init_lr, i)

        for p in meta_processes:
            for seq_name, vis_data in p['return_dict']['vis_data_seqs'].items():
                # Visdom throws no error for infinte or NaN values
                assert not np.isnan(np.array(vis_data)).any() and \
                    np.isfinite(np.array(vis_data)).all()

                vis_dict[f"{seq_name}_model_metrics"].reset()
                for epoch, vis_datum in enumerate(vis_data):
                    vis_dict[f"{seq_name}_model_metrics"].plot(vis_datum, epoch + 1)

        # optimize meta_optim
        meta_optim.zero_grad()
        for name, param in meta_optim.named_parameters():
            param.grad = meta_optim_param_grad[name] / len(db_train.seqs_dict)

            grad_clip = meta_optim_optim_cfg['grad_clip']
            if grad_clip is not None:
                param.grad.clamp_(-1.0 * grad_clip, grad_clip)

        meta_optim_optim.step()

        # join and visualize evaluation
        for dataset_name, p in eval_processes.items():
            if not p['process'].is_alive():
                p['process'].join()

                J_seq_vis = [torch.tensor(p['return_dict']['init_J_seq']).mean(),
                            torch.tensor(p['return_dict']['J_seq']).mean()]
                J_seq_vis.extend(p['return_dict']['J_seq'])
                vis_dict[f'{dataset_name}_J_seq_vis'].plot(J_seq_vis, p['num_meta_run'])

        # save_model_to_db(meta_optim, f"update_{i}.model", ex)
        # torch.save(meta_optim, os.path.join(save_dir, 'meta', f"meta_run_{i + 1}.model"))
