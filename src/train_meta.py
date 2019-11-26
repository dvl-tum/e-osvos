import copy
import math
import os
import logging
import shutil
import socket
import tempfile
import timeit
from datetime import datetime
from itertools import chain, count

import imageio
import networks.vgg_osvos as vo
import numpy as np
import sacred
import torch
import torch.multiprocessing as mp
from torchvision import transforms
from meta_optim.meta_optim import MetaOptimizer
from meta_optim.utils import dict_to_html
from pytorch_tools.ingredients import (save_model_to_db, set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import LineVis, TextVis
from spatial_correlation_sampler import spatial_correlation_sample
from torch.utils.data import DataLoader
from data import custom_transforms
from util.helper_func import (compute_loss, data_loaders, early_stopping,
                              epoch_iter, eval_davis_seq, eval_loader, grouper,
                              init_parent_model, run_loader, train_val, update_dict)

mp.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

torch_ingredient.add_config('cfgs/torch.yaml')

ex = sacred.Experiment('osvos-meta', ingredients=[torch_ingredient])
ex.add_config('cfgs/meta.yaml')
ex.add_named_config('DAVIS-17', 'cfgs/meta_davis-17.yaml')
ex.add_named_config('YouTube-VOS', 'cfgs/meta_youtube-vos.yaml')
ex.add_named_config('cross-entropy', 'cfgs/meta_cross-entropy.yaml')


MetaOptimizer = ex.capture(MetaOptimizer, prefix='meta_optim_cfg')
data_loaders = ex.capture(data_loaders, prefix='data_cfg')
early_stopping = ex.capture(early_stopping, prefix='train_early_stopping')
train_val = ex.capture(train_val)


@ex.capture
def init_vis(env_suffix: str, _config: dict, _run: sacred.run.Run,
             torch_cfg: dict, datasets: dict, resume_meta_run_epoch: int):
    run_name = f"{_run.experiment_info['name']}_{env_suffix}"
    vis_dict = {}

    resume  = False if resume_meta_run_epoch is None else True

    opts = dict(title=f"CONFIG and NON META BASELINE (RUN: {_run._id})",
                width=500, height=1750)
    vis_dict['config_vis'] = TextVis(opts, env=run_name, **torch_cfg['vis'])
    if not resume:
        vis_dict['config_vis'].plot(dict_to_html(_config))

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
        xlabel='META ITERS',
        ylabel='METRICS',
        width=750,
        height=300,
        legend=legend)
    vis_dict['meta_metrics_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            loader, *_ = data_loaders(dataset)  # pylint: disable=E1120
            legend = ['INIT MEAN', 'MEAN']
            legend += [f"INIT {seq_name}" for seq_name in loader.dataset.seqs_names]
            legend += [f"{seq_name}" for seq_name in loader.dataset.seqs_names]
            opts = dict(
                title=f"EVAL: {dataset_name.upper()} J (RUN: {_run._id})",
                xlabel='META EPOCHS',
                ylabel=f'{dataset_name.upper()} J',
                width=750,
                height=750,
                legend=legend)
            vis_dict[f'{dataset_name}_J_seq_vis'] = LineVis(
                opts, env=run_name, resume=resume, **torch_cfg['vis'])

    train_loader, *_ = data_loaders(datasets['train'])  # pylint: disable=E1120

    legend = ['MEAN'] + [f"{seq_name}" for seq_name in train_loader.dataset.seqs_names]
    opts = dict(
        title=f"META LOSS (RUN: {_run._id})",
        xlabel='META EPOCHS',
        ylabel=f'META LOSS',
        width=750,
        height=750,
        legend=legend)
    vis_dict[f'meta_loss_seq_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    legend = ['TRAIN loss', 'META loss', 'LR MEAN', 'LR STD',]
            #   'LR MOM MEAN', 'WEIGHT DECAY MEAN']
    for seq_name in train_loader.dataset.seqs_names:
        opts = dict(
            title=f"SEQ METRICS - {seq_name}",
            xlabel='EPOCHS',
            width=450,
            height=300,
            legend=legend)
        vis_dict[f"{seq_name}_model_metrics"] = LineVis(
            opts, env=run_name, resume=resume, **torch_cfg['vis'])

    model, _ = init_parent_model(**_config['parent_model'])
    legend = ['MEAN', 'STD'] + [f"{n}"
              for n, p in model.named_parameters()
              if p.requires_grad]
    opts = dict(
        title=f"INIT LR (RUN: {_run._id})",
        xlabel='META ITERS',
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

    # TODO: refactor
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
        _, _, h, w = outputs.size()

        scaled_train_gts = torch.nn.functional.interpolate(train_gts, (h, w))
        match_embedding = outputs[:-1]
        train_embedding = outputs[-1:].repeat(match_embedding.size(0), 1, 1, 1)

        # l2 normalization
        match_embedding_norm = torch.norm(match_embedding, p=2, dim=1).detach()
        match_embedding = match_embedding.div(match_embedding_norm.expand_as(match_embedding))

        train_embedding_norm = torch.norm(train_embedding, p=2, dim=1).detach()
        train_embedding = train_embedding.div(train_embedding_norm.expand_as(train_embedding))

        # foreground/background
        train_foreground_embedding = train_embedding * scaled_train_gts
        train_background_embedding = train_embedding * (1 - scaled_train_gts)

        # patch_size = (h // 4, w // 4)
        # stride = 2
        # patch_size = (h * 2, w * 2)
        corr_kwargs = {'patch_size': (h, w),
                       'stride': 1,
                       'kernel_size': 1,
                       'padding': 0}

        corr_foreground = spatial_correlation_sample(
            match_embedding.cpu(), train_foreground_embedding.cpu(), **corr_kwargs)
        # corr_foreground_mean = corr_foreground.view(corr_foreground.size(0), -1).mean(dim=1, keepdim=True)
        # corr_foreground_mean /= scaled_train_gts.sum()
        corr_foreground = corr_foreground.max(dim=1)[0]
        corr_foreground = corr_foreground.max(dim=1)[0]
        corr_foreground_mean = corr_foreground#.view(corr_foreground.size(0), -1)
        # corr_foreground_mean /= scaled_train_gts.sum() / scaled_train_gts.numel()

        corr_background = spatial_correlation_sample(
            match_embedding.cpu(), train_background_embedding.cpu(), **corr_kwargs)
        corr_background = corr_background.max(dim=1)[0]
        corr_background = corr_background.max(dim=1)[0]
        corr_background_mean = corr_background#.view(corr_background.size(0), -1)

        # print(scaled_train_gts.cpu().shape,
        #       corr_foreground_mean.shape, corr_background_mean.shape)
        match_embed = torch.cat(
            [scaled_train_gts.cpu().squeeze(dim=1), corr_foreground_mean, corr_background_mean], dim=0)
        # match_embed = match_embed.to(train_embedding.device)
        match_embed = match_embed.to(train_embedding.device)

        # print(match_embed.mean(dim=1), match_embed.shape)

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


def load_state_dict(model, seq_name, parent_states):
    """
    If multiple train splits return parent model state dictionary with seq_name
    not in the training but validation split.
    """
    if parent_states['states']:
        state_dict = None
        for state, split in zip(parent_states['states'], parent_states['splits']):
            if seq_name in split:
                state_dict = state
                break
        assert state_dict is not None, \
            f'No parent model with {seq_name} in corresponding val_split_file.'
        model.load_state_dict(state_dict)


def meta_run(i: int, rank: int, samples: list, meta_optim_state_dict: dict,
             meta_optim_param_grad: dict, global_rng_state: torch.ByteTensor,
             _config: dict, dataset: str, return_dict: dict):
    loss_func = _config['loss_func']

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_random_seeds(_config['seed'])

    if _config['eval_datasets']:
        gpu_rank = (rank // _config['num_meta_processes_per_gpu']) + 1
        # datasets = {k: v for k, v in _config['datasets'].items() if v is not None}
        # device = torch.device(f"cuda:{rank + len(datasets)}")
        # meta_device = torch.device(f"cuda:{rank + len(datasets)}")
    else:
        gpu_rank = rank // _config['num_meta_processes_per_gpu']
    device = torch.device(f'cuda:{gpu_rank}')
    meta_device = torch.device(f'cuda:{gpu_rank}')

    train_loader, _, meta_loader = data_loaders(dataset, **_config['data_cfg'])

    model, parent_states = init_parent_model(**_config['parent_model'])
    meta_model, _ = init_parent_model(**_config['parent_model'])

    meta_optim = MetaOptimizer(model, meta_model, **_config['meta_optim_cfg'])
    meta_optim.load_state_dict(meta_optim_state_dict)

    model.to(device)
    meta_model.to(meta_device)
    meta_optim.to(meta_device)

    seqs_metrics = ['train_loss', 'meta_loss', 'loss', 'J', 'F']
    seqs_metrics = {m: {s['seq_name']: [] for s in samples} for m in seqs_metrics}

    vis_data_seqs = {}

    for sample in samples:
        seq_name = sample['seq_name']
        train_loader.dataset.set_seq(seq_name)
        train_loader.dataset.frame_id = sample['train_frame_id']
        train_loader.dataset.multi_object_id = sample['multi_object_id']

        meta_loader.dataset.set_seq(seq_name)
        meta_loader.dataset.frame_id = sample['meta_frame_id']
        meta_loader.dataset.multi_object_id = sample['multi_object_id']
        meta_loader.dataset.transform = sample['meta_transform']

        vis_data_seqs[seq_name] = []

        bptt_loss = 0
        stop_train = False
        prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
        train_loss_hist = []

        meta_optim_param_grad_seq = {name: torch.zeros_like(param)
                                    for name, param in meta_optim.named_parameters()}

        load_state_dict(model, seq_name, parent_states['train'])
        meta_optim.load_state_dict(meta_optim_state_dict)
        meta_optim.reset()

        meta_optim_optim = torch.optim.Adam(meta_optim.parameters(),
                                            lr=_config['meta_optim_optim_cfg']['lr'])
        # meta_optim_optim.load_state_dict(meta_optim_optim_state_dict)

        # meta run sets its own random state for the first frame data augmentation
        # we use the global rng_state for global randomizations
        # torch.set_rng_state(global_rng_state)
        # global_rng_state = torch.get_rng_state()

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
                model.train()
                train_outputs = model(train_inputs)

                train_loss = compute_loss(loss_func, train_outputs[-1], train_gts)

                train_loss_hist.append(train_loss.item())
                train_loss.backward()

                if _config['meta_optim_cfg']['gt_input']:
                    meta_optim.train_frame_id = train_loader.dataset.frame_id
                    meta_optim.meta_frame_id = meta_loader.dataset.frame_id
                    meta_optim.seq_id = meta_loader.dataset.get_seq_id()

                meta_optim.set_train_loss(train_loss)
                meta_model, stop_train = meta_optim.step()

                meta_model.train()

                bptt_iter_loss = 0.0
                for meta_batch in meta_loader:
                    meta_inputs, meta_gts = meta_batch['image'], meta_batch['gt']
                    meta_inputs, meta_gts = meta_inputs.to(
                        meta_device), meta_gts.to(meta_device)

                    meta_outputs = meta_model(meta_inputs)
                    meta_loss = compute_loss(loss_func, meta_outputs[-1], meta_gts)
                    bptt_iter_loss += meta_loss

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

                        if ('model_init' in name and
                            _config['learn_model_init_only_from_multi_object_seqs']):
                            if train_loader.dataset.num_objects > 1:
                                meta_optim_param_grad_seq[name] += param.grad.clone()
                        else:
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
            meta_optim_param_grad[name] += grad.cpu()# / _config['num_frame_pairs_per_seq']

    # # compute mean over num_frame_pairs_per_seq
    # seqs_metrics = {metric_name: {seq_name: torch.tensor(vv).mean()
    #                               for seq_name, vv in v.items()}
    #                 for metric_name, v in seqs_metrics.items()}

    return_dict['seqs_metrics'] = seqs_metrics
    return_dict['vis_data_seqs'] = {} # vis_data_seqs
    return_dict['global_rng_state'] = global_rng_state


def evaluate(rank: int, dataset_key: str, meta_optim_state_dict: dict, _config: dict, return_dict: dict):
    seed = _config['seed']
    loss_func = _config['loss_func']
    datasets = _config['datasets']

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_random_seeds(seed)
    device = torch.device(f'cuda:{rank}')

    model, parent_states = init_parent_model(**_config['parent_model'])
    meta_model, _ = init_parent_model(**_config['parent_model'])

    meta_optim = MetaOptimizer(model, meta_model, **_config['meta_optim_cfg'])
    meta_optim.load_state_dict(meta_optim_state_dict)

    model.to(device)
    meta_optim.to(device)

    meta_model.to('cpu')

    train_loader, test_loader, meta_loader = data_loaders(  # pylint: disable=E1120
        datasets[dataset_key], **_config['data_cfg'])

    def early_stopping_func(loss_hist):
        return early_stopping(loss_hist, **_config['train_early_stopping_cfg'])

    if _config['eval_online_adapt_step'] is not None:
        assert _config['meta_optim_cfg']['matching_input'], (
            'Online adaptation must have meta_optim_cfg.matching_input=True. ')

    # save predictions in human readable format
    if _config['save_eval_preds']:
        preds_save_dir = os.path.join(f"log/eval/{_config['env_suffix']}")
        if not os.path.exists(preds_save_dir):
            os.makedirs(preds_save_dir)
        for seq_name in train_loader.dataset.seqs_names:
            if not os.path.exists(os.path.join(preds_save_dir, seq_name)):
                os.makedirs(os.path.join(preds_save_dir, seq_name))

    # temp directory for predictions for metrics evaluation
    temp_preds_save_dir = tempfile.mkdtemp()
    for seq_name in train_loader.dataset.seqs_names:
        if not os.path.exists(os.path.join(temp_preds_save_dir, seq_name)):
            os.makedirs(os.path.join(temp_preds_save_dir, seq_name))

    init_J_seq = []
    J_seq = []
    for seq_name in train_loader.dataset.seqs_names:
        train_loader.dataset.set_seq(seq_name)
        test_loader.dataset.set_seq(seq_name)
        meta_loader.dataset.set_seq(seq_name)

        # initial metrics
        # if multi object is treated as multiple single objects init J without
        # fine-tuning returns no reasonable results
        if not _config['data_cfg']['multi_object']:
            load_state_dict(model, seq_name, parent_states[dataset_key])
            meta_optim.reset()
            meta_optim.eval()
            _, _, J, _,  = eval_loader(model, test_loader, loss_func)
            init_J_seq.append(J)

        preds = []
        for obj_id in range(train_loader.dataset.num_objects):
            train_loader.dataset.multi_object_id = obj_id
            meta_loader.dataset.multi_object_id = obj_id
            test_loader.dataset.multi_object_id = obj_id

            load_state_dict(model, seq_name, parent_states[dataset_key])

            # evaluation with online adaptation
            if _config['eval_online_adapt_step'] is None:
                # one iteration with original meta frame and evaluation of entire sequence
                eval_online_adapt_step = len(test_loader.dataset)
                meta_frame_iter = [meta_loader.dataset.frame_id]
            else:
                eval_online_adapt_step = _config['eval_online_adapt_step']
                meta_frame_iter = range(eval_online_adapt_step,
                                    len(test_loader.dataset),
                                    eval_online_adapt_step)

            # meta_frame_id might be a str, e.g., 'middle'
            for i, meta_frame_id in enumerate(meta_frame_iter):
                # range [min, max[
                if i == 0:
                    # save gt of first frame as prediction of first frame
                    # TODO: refactor
                    # get train frame without random transformation
                    test_loader_frame_id = test_loader.dataset.frame_id
                    test_loader.dataset.frame_id = None
                    train_frame = test_loader.dataset[train_loader.dataset.frame_id]
                    test_loader.dataset.frame_id = test_loader_frame_id
                    train_frame_gt = train_frame['gt']

                    if not obj_id:
                        preds.append((obj_id + 1) * train_frame_gt)
                    else:
                        preds[i][train_frame_gt == 1.0] = obj_id + 1

                    eval_frame_range_min = 1
                    eval_frame_range_max = eval_online_adapt_step // 2 + 1
                else:
                    # eval_frame_range_min = (meta_frame_id - eval_online_adapt_step // 2) + 1
                    eval_frame_range_min = eval_frame_range_max

                eval_frame_range_max += eval_online_adapt_step
                if eval_frame_range_max + (eval_online_adapt_step // 2 + 1) > len(test_loader.dataset):
                    eval_frame_range_max = len(test_loader.dataset)

                load_state_dict(model, seq_name, parent_states[dataset_key])

                meta_optim.reset()
                meta_optim.eval()

                if _config['meta_optim_cfg']['matching_input']:
                    meta_loader_frame_id = meta_loader.dataset.frame_id
                    meta_loader.dataset.frame_id = meta_frame_id
                    match_embed(model, train_loader, meta_loader)
                    meta_loader.dataset.frame_id = meta_loader_frame_id

                    # for module in model.modules_with_requires_grad_params():
                    #     print(dataset_key, seq_name, module.match_embed.mean(dim=0, keepdim=True).detach().mean())

                train_val(  # pylint: disable=E1120
                    model, train_loader, None, meta_optim, _config['num_epochs'], seed,
                    early_stopping_func=early_stopping_func,
                    validate_inter=None,
                    loss_func=loss_func)

                # for epoch in epoch_iter(_config['num_epochs']):
                #     set_random_seeds(_config['seed'] + epoch)

                #     train_loss_hist = []
                #     for train_batch in train_loader:
                #         train_inputs, train_gts = train_batch['image'], train_batch['gt']
                #         train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

                #         model.train()
                #         train_outputs = model(train_inputs)

                #         train_loss = compute_loss(loss_func, train_outputs[-1], train_gts)
                #         train_loss_hist.append(train_loss.detach())

                #         model.zero_grad()
                #         train_loss.backward()

                #         meta_optim.set_train_loss(train_loss)
                #         with torch.no_grad():
                #             meta_optim.step()

                #         if early_stopping_func(train_loss_hist):
                #             break

                # run model on frame range
                test_loader.sampler.indices = range(eval_frame_range_min, eval_frame_range_max)
                _, _, preds_frame_range = run_loader(model, test_loader, loss_func, return_preds=True)
                preds_frame_range = preds_frame_range.cpu()

                for frame_id, pred in zip(test_loader.sampler.indices, preds_frame_range):
                    if not obj_id:
                        preds.append((obj_id + 1) * pred)
                    else:
                        preds[frame_id][pred == 1.0] = obj_id + 1

                test_loader.sampler.indices = None

                if eval_frame_range_max == len(test_loader.dataset):
                    break

            # break

        # TODO: refactor
        test_loader_frame_id = test_loader.dataset.frame_id
        test_loader.dataset.frame_id = None
        for frame_id, pred in enumerate(preds):
            file_name = test_loader.dataset[frame_id]['file_name']
            pred = np.transpose(pred.cpu().numpy(), (1, 2, 0)).astype(np.uint8)

            pred_path = os.path.join(temp_preds_save_dir, seq_name, os.path.basename(file_name) + '.png')
            imageio.imsave(pred_path, pred)

            if _config['save_eval_preds']:
                pred_path = os.path.join(preds_save_dir, seq_name, os.path.basename(file_name) + '.png')
                # TODO: implement color palette for labels
                imageio.imsave(pred_path, 20 * pred)
        test_loader.dataset.frame_id = test_loader_frame_id

        evaluation = eval_davis_seq(temp_preds_save_dir, seq_name)
        J = evaluation['J']['mean'][0]
        J_seq.append(J)

    shutil.rmtree(temp_preds_save_dir)

    return_dict['init_J_seq'] = init_J_seq
    return_dict['J_seq'] = J_seq


@ex.capture
def generate_meta_train_tasks(datasets: dict, num_frame_pairs_per_seq: int,
                              random_meta_frame_transform_per_task: bool,
                              change_frame_ids_per_seq_epoch: dict):
    train_loader, test_loader, meta_loader = data_loaders(datasets['train'])  # pylint: disable=E1120

    # prepare mini batches for epoch. sequences and random frames.
    meta_taks = []
    for seq_name in train_loader.dataset.seqs_names:
        train_loader.dataset.set_seq(seq_name)
        meta_loader.dataset.set_seq(seq_name)
        test_loader.dataset.set_seq(seq_name)

        # for _ in range(num_frame_pairs_per_seq):
        for idx in range(len(test_loader.dataset)):
            for obj_id in range(train_loader.dataset.num_objects):
            # multi_object_id = torch.randint(train_loader.dataset.num_objects, (1,)).item()
                train_loader.dataset.multi_object_id = obj_id
                meta_loader.dataset.multi_object_id = obj_id
                meta_loader.dataset.frame_id = idx

                if change_frame_ids_per_seq_epoch['train'] == 'random':
                    raise NotImplementedError
                    train_loader.dataset.set_random_frame_id_with_label()
                elif change_frame_ids_per_seq_epoch['train'] == 'next':
                    raise NotImplementedError

                if change_frame_ids_per_seq_epoch['meta'] == 'random':
                    raise NotImplementedError
                    meta_loader.dataset.set_random_frame_id_with_label()
                elif change_frame_ids_per_seq_epoch['meta'] == 'next':
                    raise NotImplementedError

                # # ensure train and meta frames are not the same
                # if (_config['change_frame_ids_per_seq_epoch']['train'] or
                #     _config['change_frame_ids_per_seq_epoch']['meta']):
                #     if train_loader.dataset.frame_id == meta_loader.dataset.frame_id:
                #         train_loader.dataset.set_next_frame_id()
                meta_transform = meta_loader.dataset.transform
                if random_meta_frame_transform_per_task:
                    meta_transform = transforms.Compose([custom_transforms.RandomHorizontalFlip(deterministic=True),
                                                        custom_transforms.RandomScaleNRotate(rots=(-30, 30),
                                                                                            scales=(.75, 1.25),
                                                                                            deterministic=True),
                                                        custom_transforms.ToTensor(),])
                
                if meta_loader.dataset.has_frame_object():
                    meta_taks.append({'seq_name': seq_name,
                                    'train_frame_id': train_loader.dataset.frame_id,
                                    'meta_frame_id': meta_loader.dataset.frame_id,
                                    'meta_transform': meta_transform,
                                    'multi_object_id': obj_id})

    random_meta_task_idx = torch.randperm(len(meta_taks))
    return [meta_taks[i] for i in random_meta_task_idx]


@ex.automain
def main(save_train: bool, resume_meta_run_epoch: int, env_suffix: str,
         eval_datasets: bool, num_meta_processes_per_gpu: int, datasets: dict,
         meta_optim_optim_cfg: dict, meta_batch_size: int, seed: int,
         _config: dict, _log: logging):
    mp.set_start_method('spawn')

    assert datasets['train'] is not None

    set_random_seeds(seed)

    vis_dict = init_vis()  # pylint: disable=E1120

    save_dir = os.path.join(f'models/meta/{env_suffix}')
    if save_train:
        if os.path.exists(save_dir):
            if resume_meta_run_epoch is None:
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)

    if resume_meta_run_epoch is not None:
        saved_meta_run = torch.load(
            os.path.join(save_dir, f"meta_run_{resume_meta_run_epoch}.model"))
        # TODO: refactor and do in init_vis method
        for n in vis_dict.keys():
            if n in saved_meta_run['vis_win_names']:
                if saved_meta_run['vis_win_names'][n] is None:
                    vis_dict[n].removed = True
                else:
                    vis_dict[n].win = saved_meta_run['vis_win_names'][n]
            else:
                vis_dict[n].removed = True

    train_loader, _, _ = data_loaders(datasets['train'])  # pylint: disable=E1120

    #
    # processes
    #
    num_meta_processes = torch.cuda.device_count()
    if eval_datasets:
        num_meta_processes -= 1 #len(datasets)

    num_meta_processes *= num_meta_processes_per_gpu
    
    if meta_batch_size == 'full_batch':
        meta_batch_size = len(generate_meta_train_tasks())  # pylint: disable=E1120
        _log.info(f"Meta batch size is full batch: meta_batch_size={meta_batch_size}.")

    assert meta_batch_size >= num_meta_processes, ('Increase meta_batch_size to be larger than available GPUs.')
    # assert (train_loader.dataset.num_seqs * _config['num_frame_pairs_per_seq'] / meta_batch_size).is_integer()

    num_tasks_per_process = math.ceil(meta_batch_size / num_meta_processes)
    process_manager = mp.Manager()
    meta_processes = [dict() for _ in range(num_meta_processes)]

    eval_processes = {}
    if eval_datasets:
        eval_processes = {k: {} for k, v in datasets.items() if v is not None}

    #
    # Meta model
    #
    model, parent_states = init_parent_model(**_config['parent_model'])
    meta_model, _ = init_parent_model(**_config['parent_model'])

    if _config['meta_optim_cfg']['learn_model_init']:
        # must not learn model init if we work with splits on training
        if parent_states['train']['states']:
            if len(parent_states['train']['states']) > 1:
                raise NotImplementedError
            model.load_state_dict(parent_states['train']['states'][0])

    meta_optim = MetaOptimizer(model, meta_model)  # pylint: disable=E1120
    # models were only needed to setup MetaOptimizer. in this outer loop
    # the MetaOptimizer is only updated and never applied.
    del model, meta_model
    if resume_meta_run_epoch is not None:
        meta_optim.load_state_dict(saved_meta_run['meta_optim_state_dict'])
    _log.info(f"Meta optim model parameters: {sum([p.numel() for p in meta_optim.parameters()])}")

    # meta_optim.load_state_dict(torch.load('models/meta/debug_v3/meta_run_38.model')['meta_optim_state_dict'])

    meta_optim_params = []
    for n, p in meta_optim.named_parameters():
        if 'model_init' in n:
            lr = meta_optim_optim_cfg['model_init_lr']
        elif 'log_init_lr' in n:
            lr = meta_optim_optim_cfg['log_init_lr_lr']
        elif 'param_group_lstm_hx_init' in n or 'param_group_lstm_cx_init' in n:
            lr = meta_optim_optim_cfg['param_group_lstm_init_lr']
        else:
            lr = meta_optim_optim_cfg['lr']
        meta_optim_params.append({'params': [p], 'lr': lr})

    # from radam import RAdam
    # meta_optim_optim = RAdam(meta_optim_params, lr=meta_optim_optim_cfg['lr'])
    meta_optim_optim = torch.optim.Adam(meta_optim_params,
                                        lr=meta_optim_optim_cfg['lr'])
    if resume_meta_run_epoch is not None:
        meta_optim_optim.load_state_dict(saved_meta_run['meta_optim_optim_state_dict'])

    meta_optim_param_grad = process_manager.dict({name: torch.zeros_like(param).cpu()
                                                  for name, param in meta_optim.named_parameters()})

    global_rng_state = torch.get_rng_state()
    if resume_meta_run_epoch is not None:
        global_rng_state = saved_meta_run['global_rng_state']

    meta_epoch = 1
    if resume_meta_run_epoch is not None:
        meta_epoch = saved_meta_run['meta_epoch'] + 1
    for i in count(start=meta_epoch):
        # start train and val evaluation
        for rank, (dataset_key, p) in enumerate(eval_processes.items()):
            if 'process' not in p or not p['process'].is_alive():
                p['meta_epoch'] = i
                p['return_dict'] = process_manager.dict()
                process_args = [0, dataset_key, meta_optim.state_dict(),
                                _config, p['return_dict']]
                p['process'] = mp.Process(target=evaluate, args=process_args)
                p['process'].start()

        seqs_metrics = {'train_loss': {}, 'meta_loss': {}, 'loss': {},
                        'J': {}, 'F': {}}
        seqs_metrics = {m: {n: [] for n in train_loader.dataset.seqs_names} for m in seqs_metrics}

        meta_taks = generate_meta_train_tasks()  # pylint: disable=E1120
        meta_iters_per_epoch = math.ceil(len(meta_taks) / meta_batch_size)

        for meta_iter, meta_mini_batch in enumerate(grouper(meta_batch_size, meta_taks)):
            # filter None values from grouper
            meta_mini_batch = [s for s in meta_mini_batch if s is not None]

            start_time = timeit.default_timer()

            # set meta optim gradients to zero
            for p in meta_optim_param_grad.values():
                p.zero_()

            # start meta run processes
            task_groups = grouper(num_tasks_per_process, meta_mini_batch)
            for rank, (p, tasks_for_process) in enumerate(zip(meta_processes, task_groups)):
                # filter None values from grouper
                tasks_for_process = [n for n in tasks_for_process if n is not None]

                p['return_dict'] = process_manager.dict()

                process_args = [i, rank, tasks_for_process, meta_optim.state_dict(),
                                meta_optim_param_grad, global_rng_state,
                                _config, datasets['train'], p['return_dict']]
                p['process'] = mp.Process(target=meta_run, args=process_args)
                p['process'].start()

            # join meta run processes and update meta iter and seq metrics
            meta_iter_metrics = {}
            for p in meta_processes:
                p['process'].join()
                global_rng_state = p['return_dict']['global_rng_state']

                for metric, seqs_values in p['return_dict']['seqs_metrics'].items():
                    if metric not in meta_iter_metrics:
                        meta_iter_metrics[metric] = []
                    for seq_name, seq_values in seqs_values.items():
                        seqs_metrics[metric][seq_name].extend(seq_values)
                        meta_iter_metrics[metric].extend(seq_values)
                p['return_dict']['seqs_metrics'] = {}

            # optimize meta_optim
            meta_optim.zero_grad()
            for name, param in meta_optim.named_parameters():
                # normalize over batch
                param.grad = meta_optim_param_grad[name] / len(meta_mini_batch)

                grad_clip = meta_optim_optim_cfg['grad_clip']
                if grad_clip is not None:
                    param.grad.clamp_(-1.0 * grad_clip, grad_clip)

            meta_optim_optim.step()

            # visualize meta metrics and seq runs
            meta_iter_train_loss = torch.tensor(meta_iter_metrics['train_loss'])
            meta_iter_meta_loss = torch.tensor(meta_iter_metrics['meta_loss'])
            meta_metrics = [meta_iter_train_loss.mean(),
                            meta_iter_meta_loss.mean(),
                            meta_iter_meta_loss.std(),
                            meta_iter_meta_loss.max(),
                            meta_iter_meta_loss.min()]
            meta_metrics.append((timeit.default_timer() - start_time) / 60)
            vis_dict['meta_metrics_vis'].plot(
                meta_metrics, (i - 1) * meta_iters_per_epoch + meta_iter + 1)

            meta_init_lr = [meta_optim.log_init_lr.exp().mean(),
                            meta_optim.log_init_lr.exp().std()]
            meta_init_lr += meta_optim.log_init_lr.exp().detach().numpy().tolist()
            vis_dict['init_lr_vis'].plot(
                meta_init_lr, (i - 1) * meta_iters_per_epoch + meta_iter + 1)

            for p in meta_processes:
                for seq_name, vis_data in p['return_dict']['vis_data_seqs'].items():
                    # Visdom throws no error for infinte or NaN values
                    assert not np.isnan(np.array(vis_data)).any() and \
                        np.isfinite(np.array(vis_data)).all()

                    vis_dict[f"{seq_name}_model_metrics"].reset()
                    for epoch, vis_datum in enumerate(vis_data):
                        vis_dict[f"{seq_name}_model_metrics"].plot(vis_datum, epoch + 1)

        meta_loss_seq = [torch.tensor(list(chain.from_iterable(list(seqs_metrics['meta_loss'].values())))).mean()]
        for m_l in seqs_metrics['meta_loss'].values():
            meta_loss_seq.append(torch.tensor(m_l).mean())
        vis_dict[f'meta_loss_seq_vis'].plot(meta_loss_seq, i)

        # join and visualize evaluation
        for dataset_name, p in eval_processes.items():
            if not p['process'].is_alive():
                p['process'].join()

                J_seq_vis = [torch.tensor(p['return_dict']['init_J_seq']).mean(),
                            torch.tensor(p['return_dict']['J_seq']).mean()]
                J_seq_vis.extend(p['return_dict']['init_J_seq'])
                J_seq_vis.extend(p['return_dict']['J_seq'])
                vis_dict[f'{dataset_name}_J_seq_vis'].plot(J_seq_vis, p['meta_epoch'])

        if save_train:
            # save_model_to_db(meta_optim, f"update_{i}.model", ex)
            save_meta_run = {'meta_optim_state_dict': meta_optim.state_dict(),
                             'meta_optim_optim_state_dict': meta_optim_optim.state_dict(),
                             'global_rng_state': global_rng_state,
                             'vis_win_names': {n: v.win for n, v in vis_dict.items()},
                             'meta_epoch': i,
                             'seqs_metrics': seqs_metrics}
            torch.save(save_meta_run, os.path.join(save_dir, f"meta_run_{i}.model"))
