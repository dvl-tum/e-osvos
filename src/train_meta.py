import copy
import math
import os
import socket
import timeit
from datetime import datetime
from itertools import count
import shutil

import networks.vgg_osvos as vo
import numpy as np
import sacred
import torch
import torch.multiprocessing as mp
from meta_optim.meta_optim import MetaOptimizer
from meta_optim.utils import dict_to_html
from pytorch_tools.ingredients import (save_model_to_db, set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import LineVis, TextVis
from torch.utils.data import DataLoader
from util.helper_func import (data_loaders, early_stopping, eval_loader, grouper,
                              init_parent_model, run_loader, setup_davis_eval,
                              train_val, update_dict, compute_loss, epoch_iter)
from spatial_correlation_sampler import spatial_correlation_sample


torch_ingredient.add_config('cfgs/torch.yaml')

ex = sacred.Experiment('osvos-meta', ingredients=[torch_ingredient])
ex.add_config('cfgs/meta.yaml')


MetaOptimizer = ex.capture(MetaOptimizer, prefix='meta_optim_cfg')
data_loaders = ex.capture(data_loaders, prefix='data_cfg')
early_stopping = ex.capture(early_stopping, prefix='train_early_stopping')
init_parent_model = ex.capture(init_parent_model, prefix='parent_model')
train_val = ex.capture(train_val)


@ex.capture
def init_vis(env_suffix: str, _config: dict, _run: sacred.run.Run,
             torch_cfg: dict, datasets: dict, resume_meta_run: int):
    run_name = f"{_run.experiment_info['name']}_{env_suffix}"
    vis_dict = {}

    opts = dict(title=f"CONFIG and NON META BASELINE (RUN: {_run._id})",
                width=500, height=1750)
    vis_dict['config_vis'] = TextVis(opts, env=run_name, **torch_cfg['vis'])
    vis_dict['config_vis'].plot(dict_to_html(_config))

    resume  = False if resume_meta_run is None else True

    legend = [
        'MEAN seq TRAIN loss',
        'MEAN seq META loss',
        'STD seq META loss',
        'MAX seq META loss',
        'MIN seq META loss',
        # 'MEAN seq loss',
        # 'MEAN seq J',
        # 'MEAN seq F',
        'RUN TIME [min]']
    opts = dict(
        title=f"TRAIN METRICS (RUN: {_run._id})",
        xlabel='NUM META RUNS',
        ylabel='FINAL METRICS',
        width=750,
        height=300,
        legend=legend)
    vis_dict['meta_metrics_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    for dataset_name, dataset in datasets.items():
        loader, *_ = data_loaders(dataset)  # pylint: disable=E1120
        legend = ['INIT MEAN', 'MEAN'] + [f"{seq_name}"
                  for seq_name in loader.dataset.seqs_dict.keys()]
        opts = dict(
            title=f"EVAL: {dataset_name.upper()} J (RUN: {_run._id})",
            xlabel='NUM META RUNS',
            ylabel=f'{dataset_name.upper()} J',
            width=750,
            height=750,
            legend=legend)
        vis_dict[f'{dataset_name}_J_seq_vis'] = LineVis(
            opts, env=run_name, resume=resume, **torch_cfg['vis'])

    train_loader, *_ = data_loaders(datasets['train'])  # pylint: disable=E1120

    legend = [f"{seq_name}" for seq_name in train_loader.dataset.seqs_dict.keys()]
    opts = dict(
        title=f"META LOSS (RUN: {_run._id})",
        xlabel='NUM META RUNS',
        ylabel=f'META LOSS',
        width=750,
        height=750,
        legend=legend)
    vis_dict[f'meta_loss_seq_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    legend = ['TRAIN loss', 'META loss', 'LR MEAN', 'LR STD',]
            #   'LR MOM MEAN', 'WEIGHT DECAY MEAN']
    for seq_name in train_loader.dataset.seqs_dict.keys():
        opts = dict(
            title=f"SEQ METRICS - {seq_name}",
            xlabel='EPOCHS',
            width=450,
            height=300,
            legend=legend)
        vis_dict[f"{seq_name}_model_metrics"] = LineVis(
            opts, env=run_name, resume=resume, **torch_cfg['vis'])

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
    vis_dict['init_lr_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    return vis_dict


def match_embed(model: torch.nn.Module, train_loader: DataLoader,
                match_loader: DataLoader):
    """
        Adds match_emb attribute to each param group of model. This attribute is
        then used by the meta optimzier.
    """
    device = next(model.parameters()).device
    model.eval()

    for module in model.modules_with_requires_grad_params():
        module.meta_embed = None
        module.match_embed = None
    
    # get train frame without random transformation
    match_loader_frame_id = match_loader.dataset.frame_id
    match_loader.dataset.frame_id = None
    train_batch = match_loader.dataset[train_loader.dataset.frame_id]
    match_loader.dataset.frame_id = match_loader_frame_id

    train_inputs, train_gts = train_batch['image'], train_batch['gt']
    train_inputs = train_inputs.to(device).unsqueeze(dim=0)
    train_gts = train_gts.to(device).unsqueeze(dim=0)

    # def _accum_meta_embed(self, inputs, outputs):
    #     batch, feat, *_ = outputs.size()
        
    #     if self.meta_embed is None:
    #         self.meta_embed = outputs.view(batch, feat, -1).mean(dim=2)
    #     else:
    #         self.meta_embed = torch.cat(
    #             [self.meta_embed, outputs.view(batch, feat, -1).mean(dim=2)])

    # accum_meta_embed_hooks = [m.register_forward_hook(_accum_meta_embed)
    #                           for m in model.modules_with_requires_grad_params()]

    def _match_embed(self, inputs, outputs):
        if isinstance(outputs, list):
            outputs = outputs[0]
        batch, feat, h, w = outputs.size()
        
        scaled_train_gts = torch.nn.functional.interpolate(train_gts, (h, w))
        match_embed = outputs[:-1]
        train_embed = outputs[-1:].repeat(match_embed.size(0), 1, 1, 1)

        train_foreground_embed = train_embed * scaled_train_gts
        train_background_embed = train_embed * (1 - scaled_train_gts)
        
        # patch_size = (h // 4, w // 4)
        # stride = 2
        patch_size = (h * 2, w * 2)
        stride = 1

        corr_foreground = spatial_correlation_sample(
            match_embed, train_foreground_embed,
            kernel_size=3, stride=stride, padding=1, patch_size=patch_size)
        corr_foreground_mean = corr_foreground.view(corr_foreground.size(0), -1).mean(dim=1, keepdim=True)
        corr_foreground_mean /= scaled_train_gts.sum()

        corr_background = spatial_correlation_sample(
            match_embed, train_background_embed,
            kernel_size=3, stride=stride, padding=1, patch_size=patch_size)
        corr_background_mean = corr_background.view(corr_background.size(0), -1).mean(dim=1, keepdim=True)
        corr_background_mean /= (1 - scaled_train_gts).sum()

        match_embed = torch.cat([corr_foreground_mean, corr_background_mean], dim=1)
        
        if self.match_embed is None:
            self.match_embed = match_embed
        else:
            self.match_embed = torch.cat([self.match_embed, match_embed], dim=0)

    match_embed_hooks = [m.register_forward_hook(_match_embed)
                         for m in model.modules_with_requires_grad_params()]

    with torch.no_grad():
        for match_batch in match_loader:
            match_inputs = match_batch['image'].to(device)
            
            inputs = torch.cat([match_inputs, train_inputs])
            model(inputs)

    for hook in match_embed_hooks:
        hook.remove()
    
    # def _match_embed(self, inputs, outputs):
    #     batch, feat, *_ = outputs.size()
    #     train_embed = outputs.view(batch, feat, -1).mean(dim=2).detach()
    #     self.match_embed = train_embed.sub(self.meta_embed).abs().mean() * 100

    # match_embed_hooks = [m.register_forward_hook(_match_embed)
    #                      for m in model.modules_with_requires_grad_params()]

    # with torch.no_grad():
    #     # get train frame without random transformation
    #     match_loader_frame_id = match_loader.dataset.frame_id
    #     match_loader.dataset.frame_id = None
        
    #     batch = match_loader.dataset[train_loader.dataset.frame_id]
    #     inputs, gts = batch['image'], batch['gt']
    #     inputs, gts = inputs.to(device).unsqueeze(dim=0), gts.to(device).unsqueeze(dim=0)

    #     mean_img = inputs.mean().expand_as(inputs)
    #     foreground_inputs = mean_img * (1 - gts) + inputs * gts
    #     model(foreground_inputs)
        
    #     match_loader.dataset.frame_id = match_loader_frame_id

    # for hook in match_embed_hooks:
    #     hook.remove()


def meta_run(i: int, rank: int, seq_names: list, meta_optim_state_dict: dict,
             meta_optim_param_grad: dict, random_frame_rng_state: torch.ByteTensor,
             _config: dict, dataset: str, return_dict: dict):
    loss_func = _config['loss_func']

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_random_seeds(_config['seed'])

    setup_davis_eval(_config['data_cfg'])

    # device = torch.device(f'cuda:{(2 * rank) + 1}')
    # meta_device = torch.device(f'cuda:{(2 * rank + 1) + 1}')
    if _config['eval_datasets']:
        device = torch.device(f'cuda:{rank + 1}')
        meta_device = torch.device(f'cuda:{rank + 1}')
    else:
        device = torch.device(f'cuda:{rank}')
        meta_device = torch.device(f'cuda:{rank}')

    train_loader, _, meta_loader = data_loaders(dataset, **_config['data_cfg'])

    model, parent_states = init_parent_model(**_config['parent_model'])
    meta_model, _ = init_parent_model(**_config['parent_model'])

    meta_optim = MetaOptimizer(model, meta_model, **_config['meta_optim_cfg'])
    meta_optim.load_state_dict(meta_optim_state_dict)

    model.to(device)
    meta_model.to(meta_device)
    meta_optim.to(meta_device)

    seqs_metrics = ['train_loss', 'meta_loss', 'loss', 'J', 'F']
    seqs_metrics = {m: {n: [] for n in seq_names} for m in seqs_metrics}

    vis_data_seqs = {}

    for _ in range(_config['num_seq_epochs_per_meta_run']):
        for seq_name in seq_names:
            train_loader.dataset.set_seq(seq_name)
            meta_loader.dataset.set_seq(seq_name)

            vis_data_seqs[seq_name] = []

            bptt_loss = 0
            stop_train = False
            prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
            train_loss_hist = []

            meta_optim_param_grad_seq = {name: torch.zeros_like(param)
                                        for name, param in meta_optim.named_parameters()}

            for state, split in zip(parent_states['train']['states'], parent_states['train']['splits']):
                if seq_name in split:
                    model.load_state_dict(state)
                    break
            meta_optim.load_state_dict(meta_optim_state_dict)
            meta_optim.reset()

            meta_optim_optim = torch.optim.Adam(meta_optim.parameters(),
                                                lr=_config['meta_optim_optim_cfg']['lr'])
            # meta_optim_optim.load_state_dict(meta_optim_optim_state_dict)

            torch.set_rng_state(random_frame_rng_state)
            
            # TODO> refactor
            if _config['change_frame_ids_per_seq_epoch']['train'] == 'random':
                train_loader.dataset.set_random_frame_id()
            elif _config['change_frame_ids_per_seq_epoch']['train'] == 'next':
                train_loader.dataset.set_next_frame_id()

            if _config['change_frame_ids_per_seq_epoch']['meta'] == 'random':
                meta_loader.dataset.set_random_frame_id()
            elif _config['change_frame_ids_per_seq_epoch']['meta'] == 'next':
                meta_loader.dataset.set_next_frame_id()
            
            if _config['change_frame_ids_per_seq_epoch']['train'] == 'random' or _config['change_frame_ids_per_seq_epoch']['meta'] == 'random':
                # ensure train and meta frame ids are not the same
                if train_loader.dataset.frame_id == meta_loader.dataset.frame_id:
                    if train_loader.dataset.frame_id + 1 == len(train_loader.dataset.img_list):
                        meta_loader.dataset.frame_id -= 1    
                    else:
                        meta_loader.dataset.frame_id += 1

            random_frame_rng_state = torch.get_rng_state()
            
            if _config['meta_optim_cfg']['matching_input']:
                match_embed(model, train_loader, meta_loader)

            for epoch in epoch_iter(_config['num_epochs']):
                if _config['increase_seed_per_meta_run']:
                    set_random_seeds(_config['seed'] + epoch + i)
                else:
                    set_random_seeds(_config['seed'] + epoch)

                for train_batch in train_loader:
                    train_inputs, train_gts = train_batch['image'], train_batch['gt']
                    train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

                    model.zero_grad()
                    model.train_no_batch_norm()
                    train_outputs = model(train_inputs)
                    
                    train_loss = compute_loss(loss_func, train_outputs[-1], train_gts)
                    train_loss_hist.append(train_loss.item())
                    train_loss.backward()

                    meta_optim.set_train_loss(train_loss)
                    meta_model, stop_train = meta_optim.step()

                    meta_model.train_no_batch_norm()

                    bptt_iter_loss = 0.0
                    for meta_batch in meta_loader:
                        meta_inputs, meta_gts = meta_batch['image'], meta_batch['gt']
                        meta_inputs, meta_gts = meta_inputs.to(
                            meta_device), meta_gts.to(meta_device)

                        meta_outputs = meta_model(meta_inputs)
                        bptt_iter_loss += compute_loss(loss_func, meta_outputs[-1], meta_gts)

                    bptt_loss += bptt_iter_loss - prev_bptt_iter_loss
                    prev_bptt_iter_loss = bptt_iter_loss.detach()

                    # visualization
                    lr = meta_optim.state["log_lr"].exp()
                    # lr_mom = meta_optim.state["lr_mom_logit"].sigmoid()
                    # weight_decay = meta_optim.state["log_weight_decay"].exp()

                    vis_data = [train_loss.item(), bptt_loss.item(),
                                lr.mean().item(), lr.std().item(),]
                                # lr_mom.mean().item(), weight_decay.mean().item()]
                    vis_data_seqs[seq_name].append(vis_data)

                    stop_train = stop_train or early_stopping(
                        train_loss_hist, **_config['train_early_stopping_cfg'])

                    # update params of meta optim
                    if not epoch % _config['bptt_epochs'] or stop_train or epoch == _config['num_epochs']:
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
                meta_optim_param_grad[name] += grad.cpu() / _config['num_seq_epochs_per_meta_run']

    # # compute mean over num_seq_epochs_per_meta_run
    # seqs_metrics = {metric_name: {seq_name: torch.tensor(vv).mean()
    #                               for seq_name, vv in v.items()}
    #                 for metric_name, v in seqs_metrics.items()}

    return_dict['seqs_metrics'] = seqs_metrics
    return_dict['vis_data_seqs'] = vis_data_seqs
    return_dict['random_frame_rng_state'] = random_frame_rng_state


def evaluate(rank: int, dataset_key: str, datasets: dict, meta_optim_state_dict: dict, _config: dict, return_dict: dict):
    seed = _config['seed']
    loss_func = _config['loss_func']

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_random_seeds(seed)
    device = torch.device(f'cuda:0')

    setup_davis_eval(_config['data_cfg'])

    model, parent_states = init_parent_model(**_config['parent_model'])
    meta_model, _ = init_parent_model(**_config['parent_model'])

    meta_optim = MetaOptimizer(model, meta_model, **_config['meta_optim_cfg'])
    meta_optim.load_state_dict(meta_optim_state_dict)

    model.to(device)
    meta_model.to(device)
    meta_optim.to(device)

    train_loader, test_loader, _ = data_loaders(datasets[dataset_key], **_config['data_cfg'])  # pylint: disable=E1120

    def early_stopping_func(loss_hist):
        return early_stopping(loss_hist, **_config['train_early_stopping_cfg'])

    init_J_seq = []
    J_seq = []
    for seq_name in train_loader.dataset.seqs_dict.keys():
        train_loader.dataset.set_seq(seq_name)
        test_loader.dataset.set_seq(seq_name)

        for state, split in zip(parent_states[dataset_key]['states'], parent_states[dataset_key]['splits']):
            if seq_name in split:
                model.load_state_dict(state)
                break

        meta_optim.reset()
        meta_optim.eval()

        _, _, J, _, _ = eval_loader(model, test_loader, loss_func, return_preds=True)
        init_J_seq.append(J)

        if _config['meta_optim_cfg']['matching_input']:
            match_embed(model, train_loader, test_loader)

        # train_val(  # pylint: disable=E1120
        #     model, train_loader, None, meta_optim, _config['num_epochs'], seed,
        #     early_stopping_func=early_stopping_func,
        #     validate_inter=None,
        #     loss_func=loss_func)

        for epoch in epoch_iter(_config['num_epochs']):
            set_random_seeds(_config['seed'] + epoch)

            train_loss_hist = []
            for train_batch in train_loader:
                train_inputs, train_gts = train_batch['image'], train_batch['gt']
                train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

                model.train_no_batch_norm()
                train_outputs = model(train_inputs)

                train_loss = compute_loss(loss_func, train_outputs[-1], train_gts)
                train_loss_hist.append(train_loss.detach())

                model.zero_grad()
                train_loss.backward()

                meta_optim.set_train_loss(train_loss)
                with torch.no_grad():
                    meta_optim.step()

                if early_stopping_func(train_loss_hist):
                    break

        _, _, J, _ = eval_loader(model, test_loader, loss_func)
        J_seq.append(J)

    return_dict['init_J_seq'] = init_J_seq
    return_dict['J_seq'] = J_seq


@ex.automain
def main(save_train: bool, resume_meta_run: int, env_suffix: str,
         eval_datasets: bool, num_processes: int, datasets: dict,
         meta_optim_optim_cfg: dict, seed: int, _config: dict):
    mp.set_start_method("spawn")

    set_random_seeds(seed)

    vis_dict = init_vis()  # pylint: disable=E1120

    save_dir = os.path.join(f'models/meta/{env_suffix}')
    if save_train:
        if os.path.exists(save_dir):
            if resume_meta_run is None:
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)

    if resume_meta_run is not None:
        saved_meta_run = torch.load(
            os.path.join(save_dir, f"meta_run_{resume_meta_run}.model"))
        for n, win_name in saved_meta_run['vis_win_names'].items():
            vis_dict[n].win = win_name

    train_loader, *_ = data_loaders(datasets['train'])  # pylint: disable=E1120

    #
    # processes
    #
    seqs_per_process = math.ceil(len(train_loader.dataset.seqs_dict) / num_processes)
    process_manager = mp.Manager()
    meta_processes = [dict() for _ in range(num_processes)]

    eval_processes = {}
    if eval_datasets:
        eval_processes = {n: {} for n in datasets.keys()}

    #
    # Meta model
    #
    model, _ = init_parent_model()
    meta_model, _ = init_parent_model()

    if _config['meta_optim_cfg']['learn_model_init']:
        raise NotImplementedError
        # model.load_state_dict(parent_state_dict)
        # meta_model.load_state_dict(parent_state_dict)

    meta_optim = MetaOptimizer(model, meta_model)  # pylint: disable=E1120
    if resume_meta_run is not None:
        meta_optim.load_state_dict(saved_meta_run['meta_optim_state_dict'])

    del model
    del meta_model

    # TODO: refactor
    meta_optim_params = [{'params': [p for n, p in meta_optim.named_parameters() if 'model_init' in n],
                          'lr': meta_optim_optim_cfg['model_init_lr']},
                         {'params': [p for n, p in meta_optim.named_parameters() if 'log_init_lr' in n],
                          'lr': meta_optim_optim_cfg['log_init_lr_lr']},
                         {'params': [p for n, p in meta_optim.named_parameters() if 'model_init' not in n and 'log_init_lr' not in n],
                          'lr': meta_optim_optim_cfg['lr']}]
    meta_optim_optim = torch.optim.Adam(meta_optim_params,
                                        lr=meta_optim_optim_cfg['lr'])
    if resume_meta_run is not None:
        meta_optim_optim.load_state_dict(saved_meta_run['meta_optim_optim_state_dict'])

    seqs_metrics = {'train_loss': {}, 'meta_loss': {}, 'loss': {}, 'J': {}, 'F': {}}
    meta_optim_param_grad = process_manager.dict({name: torch.zeros_like(param).cpu()
                                                  for name, param in meta_optim.named_parameters()})

    random_frame_rng_state = torch.get_rng_state()
    if resume_meta_run is not None:
        random_frame_rng_state = saved_meta_run['random_frame_rng_state']

    num_meta_run = 1
    if resume_meta_run is not None:
        num_meta_run = saved_meta_run['num_meta_run'] + 1
    for i in count(start=num_meta_run):
        start_time = timeit.default_timer()

        # start train and val evaluation
        for rank, (dataset_key, p) in enumerate(eval_processes.items()):
            if 'process' not in p or not p['process'].is_alive():
                p['num_meta_run'] = i
                p['return_dict'] = process_manager.dict()
                process_args = [rank, dataset_key, datasets, meta_optim.state_dict(),
                                _config, p['return_dict']]
                p['process'] = mp.Process(target=evaluate, args=process_args)
                p['process'].start()

        # set meta optim gradients to zero
        for p in meta_optim_param_grad.values():
            p.zero_()

        # start meta run processes
        groups = grouper(seqs_per_process, train_loader.dataset.seqs_dict.keys())
        for rank, (p, seq_names) in enumerate(zip(meta_processes, groups)):
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
        # meta_metrics = [torch.tensor(list(m.values())).mean()
        #                 for m in seqs_metrics.values()]
        train_loss = torch.tensor(list(seqs_metrics['train_loss'].values()))
        meta_loss = torch.tensor(list(seqs_metrics['meta_loss'].values()))
        meta_metrics = [train_loss.mean(),
                        meta_loss.mean(),
                        meta_loss.std(),
                        meta_loss.max(),
                        meta_loss.min()]
        meta_metrics.append((timeit.default_timer() - start_time) / 60)
        vis_dict['meta_metrics_vis'].plot(meta_metrics, i)

        meta_loss_seq = []
        for seq_name in train_loader.dataset.seqs_dict.keys():
            meta_loss_seq.append(torch.tensor(seqs_metrics['meta_loss'][seq_name]).mean())
        vis_dict[f'meta_loss_seq_vis'].plot(meta_loss_seq, i)

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
            param.grad = meta_optim_param_grad[name] / len(train_loader.dataset.seqs_dict)

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

        if save_train:
            # save_model_to_db(meta_optim, f"update_{i}.model", ex)
            save_meta_run = {'meta_optim_state_dict': meta_optim.state_dict(),
                             'meta_optim_optim_state_dict': meta_optim_optim.state_dict(),
                             'random_frame_rng_state': random_frame_rng_state,
                             'vis_win_names': {n: v.win for n, v in vis_dict.items()},
                             'num_meta_run': i}
            torch.save(save_meta_run, os.path.join(save_dir, f"meta_run_{i}.model"))

