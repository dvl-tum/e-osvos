import copy
import logging
import math
import os
import random
import shutil
import socket
import tempfile
import time
import timeit
from datetime import datetime
from itertools import chain, count

import imageio
import matplotlib.pyplot as plt
import networks.vgg_osvos as vo
import numpy as np
import sacred
import torch
import torch.multiprocessing as mp
import torchvision
from data import custom_transforms
from meta_optim.meta_optim import MetaOptimizer
from meta_optim.utils import dict_to_html
from networks.mask_rcnn import MaskRCNN
from pytorch_tools.ingredients import (save_model_to_db, set_random_seeds,
                                       torch_ingredient)
from pytorch_tools.vis import LineVis, TextVis
from radam import RAdam
# from spatial_correlation_sampler import spatial_correlation_sample
from torch.utils.data import DataLoader
from torchvision import transforms
from util.helper_func import (compute_loss, data_loaders, early_stopping,
                              epoch_iter, eval_davis_seq, eval_loader, grouper,
                              init_parent_model, run_loader, train_val,
                              update_dict)
from util.shared_optim import SharedAdam, ensure_shared_grads

# mp.set_sharing_strategy('file_system')
# mp.set_start_method('spawn')

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

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
             torch_cfg: dict, datasets: dict, resume_meta_run_epoch_mode: str):
    run_name = f"{_run.experiment_info['name']}_{datasets['train']['name']}_{env_suffix}"
    vis_dict = {}
    resume  = False if resume_meta_run_epoch_mode is None else True

    opts = dict(title=f"CONFIG and NON META BASELINE (RUN: {_run._id})",
                width=900, height=2000)
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
        'RUN TIME per ITER [min]']
    opts = dict(
        title=f"TRAIN METRICS (RUN: {_run._id})",
        xlabel='META ITERS',
        ylabel='METRICS',
        width=900,
        height=450,
        legend=legend)
    vis_dict['meta_metrics_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    for dataset_name, dataset in datasets.items():
        if dataset is not None:

            flip_labels = [False]
            if _config['random_flip_label']:
                flip_labels = [False, True]

            for flip_label in flip_labels:
                loader, *_ = data_loaders(dataset)  # pylint: disable=E1120
                legend = ['TIME PER FRAME', 'MEAN_TRAIN_LOSS']
                if _config['parent_model']['architecture'] == 'MaskRCNN':
                    legend.extend(['MEAN_TRAIN_LOSS_cls_score',
                                   'MEAN_TRAIN_LOSS_bbox_pred',
                                   'MEAN_TRAIN_LOSS_mask_fcn_logits'])
                legend.extend(['J & F MEAN', 'J MEAN', 'J RECALL MEAN', 'F MEAN', 'F RECALL MEAN', 'INIT J MEAN (SINGLE OBJ SEQS)'])
                for seq_name in loader.dataset.seqs_names:
                    loader.dataset.set_seq(seq_name)
                    if loader.dataset.num_objects == 1:
                        legend.append(f"INIT J_{seq_name}_1")

                for seq_name in loader.dataset.seqs_names:
                    loader.dataset.set_seq(seq_name)
                    legend.extend([f"J_{seq_name}_{i + 1}" for i in range(loader.dataset.num_objects)])

                opts = dict(
                    title=f"EVAL: {dataset_name.upper()} - FLIP LABEL: {flip_label} (RUN: {_run._id})",
                    xlabel='META ITERS',
                    ylabel=f'METRICS',
                    width=900,
                    height=450,
                    legend=legend)
                vis_dict[f'{dataset_name}_flip_label_{flip_label}_eval_seq_vis'] = LineVis(
                    opts, env=run_name, resume=resume, **torch_cfg['vis'])

    train_loader, *_ = data_loaders(datasets['train'])  # pylint: disable=E1120

    legend = ['MEAN']
    if _config['parent_model']['architecture'] == 'MaskRCNN':
        legend.extend(['MEAN cls_score', 'MEAN bbox_pred', 'MEAN mask_fcn_logits'])
    legend.extend([f"{seq_name}" for seq_name in train_loader.dataset.seqs_names])
    opts = dict(
        title=f"FINAL META LOSS (RUN: {_run._id})",
        xlabel='META EPOCHS',
        ylabel=f'META LOSS',
        width=900,
        height=450,
        legend=legend)
    vis_dict[f'meta_loss_seq_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    legend = ['MEAN', 'MEAN cls_score', 'MEAN bbox_pred', 'MEAN mask_fcn_logits'] + \
        [f"{seq_name}" for seq_name in train_loader.dataset.seqs_names]
    opts = dict(
        title=f"INIT TRAIN LOSS (RUN: {_run._id})",
        xlabel='META EPOCHS',
        ylabel=f'TRAIN LOSS',
        width=900,
        height=450,
        legend=legend)
    vis_dict[f'train_loss_seq_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    # legend = ['TRAIN loss', 'META loss', 'LR MEAN', 'LR STD',]
    #         #   'LR MOM MEAN', 'WEIGHT DECAY MEAN']
    # for seq_name in train_loader.dataset.seqs_names:
    #     opts = dict(
    #         title=f"SEQ METRICS - {seq_name}",
    #         xlabel='EPOCHS',
    #         width=450,
    #         height=450,
    #         legend=legend)
    #     vis_dict[f"{seq_name}_model_metrics"] = LineVis(
    #         opts, env=run_name, resume=resume, **torch_cfg['vis'])

    model, _ = init_parent_model(**_config['parent_model'])
    legend = ['MEAN', 'STD'] + [f"{n}"
              for n, p in model.named_parameters()
              if p.requires_grad]
    opts = dict(
        title=f"FIRST EPOCH INIT LR (RUN: {_run._id})",
        xlabel='META ITERS',
        ylabel='LR',
        width=900,
        height=450,
        legend=legend)
    vis_dict['init_lr_vis'] = LineVis(
        opts, env=run_name, resume=resume, **torch_cfg['vis'])

    opts = dict(
        title=f"LRS HIST (RUN: {_run._id})",
        xlabel='EPOCHS',
        ylabel='LR',
        width=900,
        height=450,
        legend=legend)
    vis_dict['lrs_hist_vis'] = LineVis(
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


@ex.capture
def device_for_process(rank: int, _config: dict):
    if _config['eval_datasets']:

        gpu_rank = (rank // _config['num_meta_processes_per_gpu'])
        if _config['gpu_per_dataset_eval']:
            datasets = {k: v for k, v in _config['datasets'].items()
                        if v is not None}
            gpu_rank += len(datasets)
        else:
            gpu_rank += 1
    else:
        gpu_rank = rank // _config['num_meta_processes_per_gpu']

    device = torch.device(f'cuda:{gpu_rank}')
    meta_device = torch.device(f'cuda:{gpu_rank}')

    return device, meta_device


# def ensure_shared_grads(model, shared_model):
#     """ working comment --- maintains the grads on the CPU """
#     for param, shared_param in zip(model.parameters(),
#                                    shared_model.parameters()):

#         # if shared_param.grad is None:
#         #     shared_param.grad = param.grad.cpu()
#         # else:
#         shared_param.grad += param.grad.cpu()


def meta_run(rank: int,
             shared_meta_optim_state_dict: dict,
             global_rng_state: torch.ByteTensor, _config: dict, dataset: str,
             shared_dict: dict, shared_variables: dict,
             shared_meta_optim_grads: dict):

    device, meta_device = device_for_process(rank, _config)

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_random_seeds(_config['seed'] + rank)

    model, parent_states = init_parent_model(**_config['parent_model'])
    if parent_states['train']['states']:
        if len(parent_states['train']['states']) > 1:
            raise NotImplementedError
        model.load_state_dict(parent_states['train']['states'][0])
    model_state_dict = model.state_dict()

    if _config['parent_model']['architecture'] == 'MaskRCNN':
        model.rpn.augment_target_proposals_mode = None
        # model.rpn.augment_target_proposals_mode = 'REPLACE'
        # model.rpn.augment_target_proposals_num_box_augs = None

    meta_optim = MetaOptimizer(model, **_config['meta_optim_cfg'])
    # meta_optim.load_state_dict(meta_optim_state_dict)

    num_epochs = _config['num_epochs']['train']

    while True:
        # main process sets iter_done=False after shared_meta_optim is updated
        while shared_dict['sub_iter_done']:
            time.sleep(0.25)
        else:
            meta_mini_batch = shared_dict['sub_meta_mini_batch']
            # filter None values from grouper
            meta_mini_batch = [s for s in meta_mini_batch if s is not None]

        # print('sub process',
        #       meta_mini_batch[0]['seq_name'],
        #       meta_mini_batch[0]['train_frame_id'],
        #       meta_mini_batch[0]['meta_frame_id'],
        #       meta_mini_batch[0]['train_transform'].transforms[1].rot)
        # exit()

        model.load_state_dict(model_state_dict)
        meta_optim.load_state_dict(shared_meta_optim_state_dict)
        meta_optim.zero_grad()

        model.to(device)
        meta_optim.to(meta_device)

        # TODO: refactor and combine seqs_metrics and vis_data_seqs
        seqs_metrics = ['train_loss', 'train_losses', 'meta_loss',
                        'meta_losses', 'loss', 'J', 'F']
        seqs_metrics = {m: {s['seq_name']: []
                            for s in meta_mini_batch}
                        for m in seqs_metrics}
        vis_data_seqs = {s['seq_name']: [] for s in meta_mini_batch}

        for sample in meta_mini_batch:
            seq_name = sample['seq_name']
            train_loader = sample['train_loader']
            meta_loader = sample['meta_loader']

            bptt_loss = 0
            stop_train = False
            prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
            train_loss_hist = []
            train_losses_hist = []
            vis_data_seqs[seq_name].append([])

            # meta_optim_param_grad_seq = {name: torch.zeros_like(param).cpu()
            #                             for name, param in meta_optim.named_parameters()}

            # load_state_dict(model, seq_name, parent_states['train'])
            # meta_optim.load_state_dict(meta_optim_state_dict)
            meta_optim.reset()

            # meta_optim_optim = torch.optim.Adam(meta_optim.parameters(),
            #                                     lr=_config['meta_optim_optim_cfg']['lr'])
            # # meta_optim_optim.load_state_dict(meta_optim_optim_state_dict)

            # meta run sets its own random state for the first frame data augmentation
            # we use the global rng_state for global randomizations
            # torch.set_rng_state(global_rng_state)
            # global_rng_state = torch.get_rng_state()

            if _config['meta_optim_cfg']['matching_input']:
                match_embed(model, train_loader, meta_loader)

            for epoch in epoch_iter(num_epochs):
                if _config['increase_seed_per_meta_run']:
                    set_random_seeds(_config['seed'] + rank + epoch + shared_variables['meta_iter'])
                else:
                    set_random_seeds(_config['seed'] + rank + epoch)

                # only single iteration
                for train_batch in train_loader:
                    train_inputs, train_gts = train_batch['image'], train_batch['gt']
                    train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

                    # model.zero_grad()
                    # model.train()
                    model.train_without_dropout()

                    if _config['parent_model']['architecture'] == 'MaskRCNN':
                        train_loss, train_losses = model(
                            train_inputs, train_gts, sample['box_coord_perm'],
                            train_loader.dataset.flip_label)
                        train_losses_hist.append({k: v.cpu().item()
                                                  for k, v in train_losses.items()})
                    else:
                        train_outputs = model(train_inputs)
                        train_loss = compute_loss(_config['loss_func'],
                                                  train_outputs[-1],
                                                  train_gts)

                    train_loss_hist.append(train_loss.item())
                    # train_loss.backward()

                meta_optim.set_train_loss(train_loss)
                if _config['meta_optim_cfg']['gt_input']:
                    meta_optim.train_frame_id = train_loader.dataset.frame_id
                    meta_optim.meta_frame_id = meta_loader.dataset.frame_id
                    meta_optim.seq_id = meta_loader.dataset.get_seq_id()
                meta_optim.step(train_loss)

                # meta_model.train()
                # meta_model.train_without_dropout()

                bptt_iter_loss = 0.0
                for meta_batch in meta_loader:
                    meta_inputs, meta_gts = meta_batch['image'], meta_batch['gt']
                    meta_inputs, meta_gts = meta_inputs.to(
                        meta_device), meta_gts.to(meta_device)

                    if _config['parent_model']['architecture'] == 'MaskRCNN':
                        meta_loss, meta_losses = model(
                            meta_inputs, meta_gts, sample['box_coord_perm'],
                            meta_loader.dataset.flip_label)
                    else:
                        meta_outputs = model(meta_inputs)
                        meta_loss = compute_loss(_config['loss_func'],
                                                 meta_outputs[-1],
                                                 meta_gts)

                    bptt_iter_loss += meta_loss

                bptt_loss += bptt_iter_loss - prev_bptt_iter_loss
                prev_bptt_iter_loss = bptt_iter_loss.detach()

                # visualization

                if meta_optim.lr_per_tensor:
                    lr = meta_optim.state["log_lr"].exp()
                else:
                    lr = torch.tensor([l.exp().mean()
                                       for l in meta_optim.state["log_lr"]])

                # lr_mom = meta_optim.state["lr_mom_logit"].sigmoid()
                # weight_decay = meta_optim.state["log_weight_decay"].exp()

                vis_data = [train_loss.item(), bptt_loss.item(), lr.cpu().detach().numpy()]
                vis_data_seqs[seq_name][-1].append(vis_data)

                stop_train = stop_train or early_stopping(
                    train_loss_hist, **_config['train_early_stopping_cfg'])

                # update params of meta optim
                if not epoch % _config['bptt_epochs'] or stop_train or epoch == num_epochs:
                    # meta_optim.zero_grad()
                    bptt_loss.backward()
                    # meta_optim_grads = torch.autograd.grad(bptt_loss,
                    #                                         meta_optim.parameters())

                    # def jacobian(y, x, create_graph=False):
                    #     jac = []
                    #     flat_y = y.reshape(-1)
                    #     grad_y = torch.zeros_like(flat_y)
                    #     for i in range(len(flat_y)):
                    #         grad_y[i] = 1.
                    #         grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
                    #         jac.append(grad_x.reshape(x.shape))
                    #         grad_y[i] = 0.
                    #     return torch.stack(jac).reshape(y.shape + x.shape)

                    # hessians = [jacobian(g, p).diag() for g, p in zip(meta_optim_grads, meta_optim.parameters())]
                    # print(meta_optim_grads[0].shape, meta_optim_grads[0].abs().sum())
                    # print(hessians[0].shape, hessians[0].abs().sum())

                    # for (name, _), grads in zip(meta_optim.named_parameters(), meta_optim_grads):
                    #     # print(name, grads.shape, grads.abs().mean(), grads.abs().max())

                    #     grad_clip = _config['meta_optim_optim_cfg']['grad_clip']
                    #     if grad_clip is not None:
                    #         grads.clamp_(-1.0 * grad_clip, grad_clip)

                    #     if ('model_init' in name and
                    #         _config['learn_model_init_only_from_multi_object_seqs']):
                    #         if train_loader.dataset.num_objects > 1:
                    #             meta_optim_param_grad[name] += grads.clone().cpu()
                    #     elif 'model_init_meta_optim_split' in parent_states:
                    #         if seq_name in parent_states['model_init_meta_optim_split']['splits'][0]:
                    #             if 'model_init' in name:
                    #                 meta_optim_param_grad[name] += grads.clone().cpu()
                    #         else:
                    #             if 'model_init' not in name:
                    #                 meta_optim_param_grad[name] += grads.clone().cpu()
                    #     else:
                    #         meta_optim_param_grad[name] += grads.clone().cpu()

                    # if _config['meta_optim_optim_cfg']['step_in_seq']:
                    #     meta_optim_optim.step()

                    meta_optim.reset(keep_state=True)
                    prev_bptt_iter_loss.zero_().detach_()
                    bptt_loss = 0

                if stop_train:
                    meta_optim.reset()
                    break

            # loss_batches, _ = run_loader(model, meta_loader, loss_func)
            # seqs_metrics['meta_loss'][seq_name].append(loss_batches.mean())
            seqs_metrics['meta_loss'][seq_name].append(meta_loss.item())
            seqs_metrics['meta_losses'][seq_name].append({k: v.cpu().item()
                                                          for k, v in meta_losses.items()})

            # loss_batches, _, J, F = eval_loader(model, test_loader, loss_func)
            # next_meta_frame_ids[seq_name] = loss_batches.argmax().item()

            # seqs_metrics['loss'][seq_name] = loss_batches.mean()
            # seqs_metrics['J'][seq_name] = J
            # seqs_metrics['F'][seq_name] = F
            # seqs_metrics['train_loss'][seq_name].append(train_loss_hist[-1])
            seqs_metrics['train_loss'][seq_name].append(train_loss_hist[0])
            if _config['parent_model']['architecture'] == 'MaskRCNN':
                seqs_metrics['train_losses'][seq_name].append(train_losses_hist[0])

        for name, param in meta_optim.named_parameters():
            shared_meta_optim_grads[name] += param.grad.cpu()

        shared_dict['seqs_metrics'] = seqs_metrics
        shared_dict['vis_data_seqs'] = vis_data_seqs
        shared_dict['global_rng_state'] = global_rng_state
        shared_dict['sub_iter_done'] = True


def evaluate(rank: int, dataset_key: str, flip_label: bool,
             shared_meta_optim_state_dict: dict, shared_variables: dict,
             _config: dict, shared_dict: dict, save_dir: str, vis_win_names: dict,):
    seed = _config['seed']
    loss_func = _config['loss_func']
    datasets = _config['datasets']

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    while True:
        meta_optim_state_dict = copy.deepcopy(shared_meta_optim_state_dict)
        meta_iter = shared_variables['meta_iter']
        meta_epoch = shared_variables['meta_epoch']

        set_random_seeds(seed)
        device = torch.device(f'cuda:{rank}')

        model, parent_states = init_parent_model(**_config['parent_model'])
        if parent_states[dataset_key]['states']:
            if len(parent_states[dataset_key]['states']) > 1:
                raise NotImplementedError
            model.load_state_dict(parent_states[dataset_key]['states'][0])

        meta_optim = MetaOptimizer(model, **_config['meta_optim_cfg'])
        meta_optim.load_state_dict(meta_optim_state_dict)

        model.to(device)
        meta_optim.to(device)

        train_loader, test_loader, meta_loader = data_loaders(  # pylint: disable=E1120
            datasets[dataset_key], **_config['data_cfg'])
        # remove random cropping
        train_loader.dataset.crop_size = None
        test_loader.dataset.crop_size = None
        meta_loader.dataset.crop_size = None

        train_loader.dataset.flip_label = flip_label
        test_loader.dataset.flip_label = flip_label
        meta_loader.dataset.flip_label = flip_label

        def early_stopping_func(loss_hist):
            return early_stopping(loss_hist, **_config['train_early_stopping_cfg'])

        if _config['eval_online_adapt_step'] is not None:
            assert _config['meta_optim_cfg']['matching_input'], (
                'Online adaptation must have meta_optim_cfg.matching_input=True. ')

        # save predictions in human readable format and with boxes
        if save_dir is not None:
            debug_preds_save_dir = os.path.join(save_dir,
                                                'best_eval_preds_debug',
                                                f"{datasets[dataset_key]['name']}",
                                                f"{datasets[dataset_key]['split']}")

            if not os.path.exists(debug_preds_save_dir):
                os.makedirs(debug_preds_save_dir)

            for seq_name in train_loader.dataset.seqs_names:
                if not os.path.exists(os.path.join(debug_preds_save_dir, seq_name)):
                    os.makedirs(os.path.join(debug_preds_save_dir, seq_name))

        # if eval_only_mode:
            # assert save_dir is not None
            preds_save_dir = os.path.join(save_dir,
                                          'best_eval_preds',
                                          f"{datasets[dataset_key]['name']}",
                                          f"{datasets[dataset_key]['split']}")
            if not os.path.exists(preds_save_dir):
                os.makedirs(preds_save_dir)
            for seq_name in train_loader.dataset.seqs_names:
                if not os.path.exists(os.path.join(preds_save_dir, seq_name)):
                    os.makedirs(os.path.join(preds_save_dir, seq_name))
        # else:
        #     # temp directory for predictions for metrics evaluation
        #     preds_save_dir = tempfile.mkdtemp()
        #     for seq_name in train_loader.dataset.seqs_names:
        #         if not os.path.exists(os.path.join(preds_save_dir, seq_name)):
        #             os.makedirs(os.path.join(preds_save_dir, seq_name))

        eval_time = 0
        num_frames = 0
        init_J_seq = []
        J_seq = []
        J_recall_seq = []
        train_loss_seq = []
        train_losses_seq = []
        F_seq = []
        F_recall_seq = []
        masks = {}
        boxes = {}
        for seq_name in train_loader.dataset.seqs_names:
            train_loader.dataset.set_seq(seq_name)
            test_loader.dataset.set_seq(seq_name)
            meta_loader.dataset.set_seq(seq_name)

            if _config['parent_model']['architecture'] == 'MaskRCNN':
                model.rpn.augment_target_proposals_mode = 'EXTEND'
                # model.rpn.augment_target_proposals_num_box_augs = 10

            # initial metrics
            # if multi object is treated as multiple single objects, init J without
            # fine-tuning returns no reasonable results
            if train_loader.dataset.num_objects == 1:
                test_loader.dataset.multi_object_id = 0

                # load_state_dict(model, seq_name, parent_states[dataset_key])
                meta_optim.load_state_dict(meta_optim_state_dict)
                meta_optim.reset()
                meta_optim.eval()

                if test_loader.dataset.test_mode:
                    J = [0.0]
                else:
                    _, _, J, _,  = eval_loader(model, test_loader, loss_func)
                init_J_seq.extend(J)
                # init_J_seq.extend([0.0])

            masks[seq_name] = []
            boxes[seq_name] = []
            for obj_id in range(train_loader.dataset.num_objects):
                train_loader.dataset.multi_object_id = obj_id
                meta_loader.dataset.multi_object_id = obj_id
                test_loader.dataset.multi_object_id = obj_id

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
                start_eval = timeit.default_timer()
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
                            masks[seq_name].append((obj_id + 1) * train_frame_gt)
                        else:
                            masks[seq_name][i][train_frame_gt == 1.0] = obj_id + 1

                        eval_frame_range_min = 1
                        eval_frame_range_max = eval_online_adapt_step // 2 + 1
                    else:
                        # eval_frame_range_min = (meta_frame_id - eval_online_adapt_step // 2) + 1
                        eval_frame_range_min = eval_frame_range_max

                    eval_frame_range_max += eval_online_adapt_step
                    if eval_frame_range_max + (eval_online_adapt_step // 2 + 1) > len(test_loader.dataset):
                        eval_frame_range_max = len(test_loader.dataset)

                    num_frames += eval_frame_range_max - eval_frame_range_min

                    # load_state_dict(model, seq_name, parent_states[dataset_key])
                    meta_optim.load_state_dict(meta_optim_state_dict)
                    meta_optim.reset()
                    meta_optim.eval()

                    if _config['meta_optim_cfg']['matching_input']:
                        meta_loader_frame_id = meta_loader.dataset.frame_id
                        meta_loader.dataset.frame_id = meta_frame_id
                        match_embed(model, train_loader, meta_loader)
                        meta_loader.dataset.frame_id = meta_loader_frame_id

                        # for module in model.modules_with_requires_grad_params():
                        #     print(dataset_key, seq_name, module.match_embed.mean(dim=0, keepdim=True).detach().mean())

                    if _config['parent_model']['architecture'] == 'MaskRCNN':
                        model.rpn.augment_target_proposals_mode = None
                        # model.rpn.augment_target_proposals_num_box_augs = 10

                    # train_val(
                    #     model, train_loader, None, meta_optim, _config['num_epochs'], seed,
                    #     early_stopping_func=early_stopping_func,
                    #     validate_inter=None,
                    #     loss_func=loss_func)

                    train_loss_hist = []
                    for epoch in epoch_iter(_config['num_epochs']['eval']):
                        set_random_seeds(_config['seed'] + epoch)

                        for _, sample_batched in enumerate(train_loader):
                            inputs, gts = sample_batched['image'], sample_batched['gt']
                            inputs, gts = inputs.to(device), gts.to(device)

                            model.train_without_dropout()

                            if isinstance(model, MaskRCNN):
                                train_loss, train_losses = model(inputs, gts)
                            else:
                                outputs = model(inputs)
                                train_loss = compute_loss(loss_func, outputs[-1], gts)

                            train_loss_hist.append(train_loss.item())

                            model.zero_grad()

                            meta_optim.set_train_loss(train_loss)
                            meta_optim.step(train_loss)

                            if early_stopping_func(train_loss_hist):
                                break

                        if early_stopping_func(train_loss_hist):
                            break

                    train_loss_seq.append(train_loss.item())

                    if _config['parent_model']['architecture'] == 'MaskRCNN':
                        train_losses_seq.append({k: v.cpu().item()
                                                 for k, v in train_losses.items()})
                        model.rpn.augment_target_proposals_mode = 'EXTEND'
                        # model.rpn.augment_target_proposals_num_box_augs = 10

                    # run model on frame range
                    test_loader.sampler.indices = range(eval_frame_range_min, eval_frame_range_max)
                    _, _, probs_frame_range, boxes_frame_range = run_loader(model, test_loader, loss_func, return_probs=True)
                    probs_frame_range = probs_frame_range.cpu()

                    for frame_id, probs, box in zip(test_loader.sampler.indices, probs_frame_range, boxes_frame_range):
                        if not obj_id:
                            masks[seq_name].append(probs)
                            boxes[seq_name].append(box.unsqueeze(dim=0))
                        else:
                            masks[seq_name][frame_id] = torch.cat([masks[seq_name][frame_id], probs])
                            boxes[seq_name][frame_id - 1] = torch.cat([boxes[seq_name][frame_id - 1], box.unsqueeze(dim=0)])

                        if obj_id == train_loader.dataset.num_objects - 1:
                            background_mask = masks[seq_name][frame_id].max(dim=0, keepdim=True)[0].lt(0.5)
                            masks[seq_name][frame_id] = masks[seq_name][frame_id].argmax(dim=0, keepdim=True).float() + 1.0
                            masks[seq_name][frame_id][background_mask] = 0.0

                    test_loader.sampler.indices = None

                    if eval_frame_range_max == len(test_loader.dataset):
                        break

                eval_time += timeit.default_timer() - start_eval

            # TODO: refactor
            # assert test_loader.dataset.frame_id is None
            test_loader_frame_id = test_loader.dataset.frame_id
            test_loader.dataset.frame_id = None
            for frame_id, mask_frame in enumerate(masks[seq_name]):
                file_name = test_loader.dataset[frame_id]['file_name']
                mask_frame = np.transpose(mask_frame.cpu().numpy(), (1, 2, 0)).astype(np.uint8)

                if flip_label:
                    mask_frame = np.logical_not(mask_frame).astype(np.uint8)

                pred_path = os.path.join(preds_save_dir, seq_name, os.path.basename(file_name) + '.png')

                # if _config['data_cfg']['full_resolution']:
                #     import cv2
                #     w, h, _ = mask_frame.shape
                #     print(mask_frame.shape)
                #     scaling_factor = 480 / h

                #     mask_frame = cv2.resize(mask_frame, dsize=(
                #         math.ceil(w * scaling_factor), 480), interpolation=cv2.INTER_CUBIC)

                imageio.imsave(pred_path, mask_frame)
            test_loader.dataset.frame_id = test_loader_frame_id

            if test_loader.dataset.test_mode:
                evaluation = {'J': {'mean': [0.0], 'recall': [0.0]},
                              'F': {'mean': [0.0], 'recall': [0.0]}}
            else:
                evaluation = eval_davis_seq(preds_save_dir, seq_name)

            # print(evaluation)
            J_seq.extend(evaluation['J']['mean'])
            J_recall_seq.extend(evaluation['J']['recall'])
            F_seq.extend(evaluation['F']['mean'])
            F_recall_seq.extend(evaluation['F']['recall'])

        # shutil.rmtree(preds_save_dir)

        mean_J = torch.tensor(J_seq).mean().item()
        if mean_J > shared_dict['best_mean_J']:
            shared_dict['best_mean_J'] = mean_J

            if save_dir is not None:
                save_meta_run = {'meta_optim_state_dict': meta_optim.state_dict(),
                                #  'meta_optim_optim_state_dict': meta_optim_optim.state_dict(),
                                 'vis_win_names': vis_win_names,
                                 'meta_iter': meta_iter,
                                 'meta_epoch': meta_epoch,}
                torch.save(save_meta_run, os.path.join(
                    save_dir, f"best_{dataset_key}_meta_iter.model"))

                test_loader_frame_id = test_loader.dataset.frame_id
                test_loader.dataset.frame_id = None
                for (seq_name, masks_seq), (_, boxes_seq) in zip(masks.items(), boxes.items()):
                    test_loader.dataset.set_seq(seq_name)
                    test_loader.dataset.multi_object_id = 0

                    for frame_id, mask_frame in enumerate(masks_seq):
                        file_name = test_loader.dataset[frame_id]['file_name']
                        mask_frame = np.transpose(mask_frame.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        if flip_label:
                            mask_frame = np.logical_not(mask_frame).astype(np.uint8)

                        pred_path = os.path.join(debug_preds_save_dir, seq_name, os.path.basename(
                            file_name) + f'_flip_label_{flip_label}.png')
                        # TODO: implement color palette for labels

                        fig = plt.figure()
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)

                        ax.imshow(mask_frame.squeeze(2), cmap='jet', vmin=0, vmax=test_loader.dataset.num_objects)

                        if frame_id:
                            for box in boxes_seq[frame_id - 1]:
                                ax.add_patch(
                                    plt.Rectangle(
                                        (box[0], box[1]),
                                        box[2] - box[0],
                                        box[3] - box[1],
                                        fill=False,
                                        linewidth=1.0,
                                    ))

                        plt.axis('off')
                        # plt.tight_layout()
                        plt.draw()
                        plt.savefig(pred_path, dpi=100)
                        plt.close()
                        # imageio.imsave(pred_path, pred)
                test_loader.dataset.frame_id = test_loader_frame_id

        shared_dict['init_J_seq'] = init_J_seq
        shared_dict['J_seq'] = J_seq
        shared_dict['J_recall_seq'] = J_recall_seq
        shared_dict['train_losses_seq'] = train_losses_seq
        shared_dict['train_loss_seq'] = train_loss_seq
        shared_dict['F_seq'] = F_seq
        shared_dict['F_recall_seq'] = F_recall_seq
        shared_dict['time_per_frame'] = eval_time / num_frames
        # set meta_iter here to signal main process that eval is finished
        shared_dict['meta_iter'] = meta_iter


@ex.capture
def generate_meta_train_tasks(datasets: dict, random_frame_transform_per_task: bool,
                              random_flip_label: bool, random_no_label: bool,
                              data_cfg: dict, single_obj_seq_mode: str,
                              random_box_coord_perm: bool):
    train_loader, test_loader, _ = data_loaders(
        datasets['train'], **data_cfg)
    test_dataset = test_loader.dataset
    seqs_names = test_dataset.seqs_names

    if single_obj_seq_mode == 'AUGMENT':
        single_obj_seqs = []
        for seq_name in seqs_names:
            test_dataset.set_seq(seq_name)
            if test_dataset.num_objects == 1:
                single_obj_seqs.append(seq_name)

    # prepare mini batches for epoch. sequences and random frames.
    random_seqs_idx = torch.randperm(len(seqs_names))
    random_seqs_names = [seqs_names[i] for i in random_seqs_idx]

    meta_tasks = []
    for seq_name in random_seqs_names:
        test_dataset.set_seq(seq_name)
        num_objects = test_dataset.num_objects

        # if num_objects == 1 and single_obj_seq_mode == 'IGNORE':
        #     continue

        # train_frame_id = 0
        # for meta_frame_id in range(len(test_dataset)):

        # num_frames = 10
        # assert len(test_dataset) >= num_frames, f"{seq_name}"
        # # for idx in np.linspace(1, len(test_dataset), num_frames, endpoint=False, dtype=int):
        # frames = np.linspace(0, len(test_dataset), num_frames, endpoint=False, dtype=int)
        # frame_combs = np.array(np.meshgrid(frames, frames)).T.reshape(-1,2).tolist()

        # num_frames = 1
        # # epsilon = 10
        # epsilon = len(test_dataset) - 1
        # # train_idxs = torch.randint(low=0 , high=len(test_dataset) - epsilon, size=(num_frames, ))
        # train_idxs = [torch.tensor(0)]
        # epsilons = torch.randint(low=1, high=epsilon + 1, size=(num_frames, ))
        # frame_combs = [(train_frame_id.item(), train_frame_id.item() + eps.item())
        #                for train_frame_id, eps in zip(train_idxs, epsilons)]

        # min_per_seq_tasks = 1
        # if not num_objects == 1 or single_obj_seq_mode == 'AUGMENT':
        #     min_per_seq_tasks = 2

        # seq_tasks = []
        # while len(seq_tasks) < min_per_seq_tasks:
        #     seq_tasks = []

        #     train_ids = torch.randint(low=0 , high=len(test_dataset), size=(1, ))
        #     # train_ids = [torch.tensor(0)]
        #     meta_ids = torch.randint(low=0 , high=len(test_dataset), size=(1, ))
        #     frame_combs = [(train_frame_id.item(), meta_frame_id.item())
        #                 for train_frame_id, meta_frame_id in zip(train_ids, meta_ids)]

        #     # for i in range(len(frame_combs)):
        #     #     meta_frame_id = frame_combs[i][1]
        #     #     frame_combs.append((meta_frame_id, meta_frame_id))

        #     for train_frame_id, meta_frame_id in frame_combs:

        #         if train_frame_id == meta_frame_id:
        #             continue

        if num_objects == 1:
            if single_obj_seq_mode == 'IGNORE':
                continue
            elif single_obj_seq_mode == 'AUGMENT':  # or single_obj_seq_mode == 'KEEP':
                seq_obj_ids = [0, 1]
            elif single_obj_seq_mode == 'KEEP':
                seq_obj_ids = [0]
            else:
                raise NotImplementedError
        else:
            # pick 2 random obj ids such that each batch contains 2 obj per sequence
            seq_obj_ids = [obj_id.item() for obj_id
                           in torch.randperm(test_dataset.num_objects)]
            seq_obj_ids = seq_obj_ids[:2]

        for obj_id in seq_obj_ids:

        # for obj_id in range(train_loader.dataset.num_objects):
        # multi_object_id = torch.randint(train_loader.dataset.num_objects, (1,)).item()

            train_loader, _, meta_loader = data_loaders(
                datasets['train'], **data_cfg)

            train_loader.dataset.set_seq(seq_name)
            meta_loader.dataset.set_seq(seq_name)

            train_loader.dataset.multi_object_id = obj_id
            # train_loader.dataset.frame_id = train_frame_id
            train_loader.dataset.set_random_frame_id_with_label()

            meta_loader.dataset.multi_object_id = obj_id
            # meta_loader.dataset.frame_id = meta_frame_id

            meta_frame_ids = [meta_loader.dataset.set_random_frame_id_with_label()
                              for _ in range(data_cfg['batch_sizes']['meta'])]
            meta_loader.dataset.frame_id = None
            meta_loader.sampler.indices = meta_frame_ids

            if num_objects == 1 and single_obj_seq_mode == 'AUGMENT':
                single_obj_seqs_ids = list(range(len(single_obj_seqs)))
                single_obj_seqs_ids.pop(single_obj_seqs.index(seq_name))

                random_other_single_obj_seq = single_obj_seqs[random.choice(
                    single_obj_seqs_ids)]

                train_loader.dataset.augment_with_single_obj_seq_key = random_other_single_obj_seq
                meta_loader.dataset.augment_with_single_obj_seq_key = random_other_single_obj_seq

            if random_frame_transform_per_task:
                # if train_frame_id == meta_frame_id:
                #     random_transform.append(custom_transforms.RandomRemoveLabelRectangle((90, 854)))

                random_transform = [custom_transforms.ColorJitter(brightness=2, hue=.1, saturation=.1,
                                                                  deterministic=True),
                # random_transform = [
                                    custom_transforms.RandomHorizontalFlip(),
                                    custom_transforms.RandomScaleNRotate(rots=(-30, 30),
                                                                         scales=(.5, 1.5)),
                                    custom_transforms.ToTensor(),]
                random_transform = transforms.Compose(random_transform)

                meta_loader.dataset.transform = random_transform

                if not data_cfg['random_train_transform']:
                    # random_transform = [custom_transforms.RandomHorizontalFlip()]

                    # random_transform.extend([custom_transforms.RandomScaleNRotate(rots=(-30, 30),
                    #                                                                 scales=(.5, 1.5)),
                    #                             custom_transforms.ToTensor(),])
                    # random_transform = transforms.Compose(random_transform)
                    train_loader.dataset.transform = random_transform

            # flip_label = False
            if random_flip_label:
                flip_label = bool(random.getrandbits(1))
                train_loader.dataset.flip_label = flip_label
                meta_loader.dataset.flip_label = flip_label

            # no_label = False
            if random_no_label:
                no_label = bool(random.getrandbits(1))
                train_loader.dataset.no_label = no_label
                meta_loader.dataset.no_label = no_label

            box_coord_perm = None
            if random_box_coord_perm:
                box_coord_perm = torch.randperm(4)

            meta_tasks.append({'seq_name': seq_name,
                               'box_coord_perm': box_coord_perm,
                               'train_loader': train_loader,
                               'meta_loader': meta_loader})

            # # check if train and meta frame have an object after random_frame_transform_per_task is applied
            # if len(torch.unique(list(train_loader)[0]['gt'])) > 1 and len(torch.unique(list(meta_loader)[0]['gt'])) > 1:
            # # if train_loader.dataset.has_frame_object() and meta_loader.dataset.has_frame_object():
            #     seq_tasks.append({'seq_name': seq_name,
            #                         'train_loader': train_loader,
            #                         'meta_loader': meta_loader})
            # else:
            #     # if one object is not on frame. remove entire frame.
            #     seq_tasks = []
            #     break

        # meta_tasks.extend(seq_tasks)

    # random_meta_task_idx = torch.randperm(len(meta_tasks))
    # return [meta_tasks[i] for i in random_meta_task_idx]

    # print([(t['seq_name'],
    #         t['train_loader'].dataset.multi_object_id,
    #         t['meta_loader'].dataset.multi_object_id,
    #         t['train_loader'].dataset.frame_id,
    #         t['meta_loader'].dataset.frame_id,
    #         t['train_loader'].dataset.augment_with_single_obj_seq_key,
    #         t['meta_loader'].dataset.augment_with_single_obj_seq_key)
    #        for t in meta_tasks])

    return meta_tasks


@ex.automain
def main(save_dir: str, resume_meta_run_epoch_mode: str, env_suffix: str,
         eval_datasets: bool, num_meta_processes_per_gpu: int, datasets: dict,
         meta_optim_optim_cfg: dict, seed: int, _config: dict, _log: logging,
         meta_optim_model_file: str, gpu_per_dataset_eval: bool,
         meta_batch_size: int, vis_interval: int, data_cfg: dict):
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')

    assert datasets['train'] is not None

    set_random_seeds(seed)

    vis_dict = init_vis()  # pylint: disable=E1120

    if save_dir is not None:
        save_dir = os.path.join(save_dir, f"{datasets['train']['name']}_{env_suffix}")
        if os.path.exists(save_dir):
            if resume_meta_run_epoch_mode is None:
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
        else:
            os.makedirs(save_dir)

    if resume_meta_run_epoch_mode is not None:
        if resume_meta_run_epoch_mode == 'LAST':
            resume_model_name = f"last_meta_epoch.model"
        elif resume_meta_run_epoch_mode == 'BEST_VAL':
            resume_model_name = f"best_val_iter.model"
        else:
            raise NotImplementedError
        saved_meta_run = torch.load(os.path.join(save_dir, resume_model_name))
        # TODO: refactor and do in init_vis method
        for n in vis_dict.keys():
            if n in saved_meta_run['vis_win_names']:
                if saved_meta_run['vis_win_names'][n] is None:
                    vis_dict[n].removed = True
                else:
                    vis_dict[n].win = saved_meta_run['vis_win_names'][n]
            else:
                vis_dict[n].removed = True

    #
    # processes
    #
    num_meta_processes = torch.cuda.device_count()
    eval_processes = {}
    if eval_datasets:
        eval_processes = [{'dataset_key': k, 'flip_label': False}
                          for k, v in datasets.items()
                          if v is not None]

        if _config['random_flip_label']:
            eval_processes.extend([{'dataset_key': k, 'flip_label': True}
                                   for k, v in datasets.items()
                                   if v is not None])

        if gpu_per_dataset_eval:
            num_meta_processes -= len(eval_processes)
        else:
            num_meta_processes -= 1

    if num_meta_processes:
        num_meta_processes *= num_meta_processes_per_gpu

        meta_tasks = generate_meta_train_tasks()  # pylint: disable=E1120
        num_meta_tasks = len(meta_tasks)
        if meta_batch_size == 'full_batch' or meta_batch_size > num_meta_tasks:
            meta_batch_size = num_meta_tasks
            _log.warning(f"Meta batch size is full batch: meta_batch_size={meta_batch_size}.")
        if data_cfg['multi_object']:
            assert not meta_batch_size % 2, (f'meta_batch_size {meta_batch_size} is not a multiple of 2 for multi object training.')
        assert not meta_batch_size % num_meta_processes, ('meta_batch_size is not a multiple of num_meta_processes.')
    else:
        _log.warning(f"EVAL modus.")

    process_manager = mp.Manager()
    meta_processes = [dict() for _ in range(num_meta_processes)]

    #
    # Meta model
    #
    model, parent_states = init_parent_model(**_config['parent_model'])

    # if _config['meta_optim_cfg']['learn_model_init']:
    #     # must not learn model init if we work with splits on training
    if parent_states['train']['states']:
        if len(parent_states['train']['states']) > 1:
            raise NotImplementedError
        model.load_state_dict(parent_states['train']['states'][0])

    meta_optim = MetaOptimizer(model)  # pylint: disable=E1120
    meta_optim.init_zero_grad()

    if meta_optim_model_file is not None:
        previous_meta_optim_state_dict = torch.load(meta_optim_model_file)['meta_optim_state_dict']
        # meta_optim_state_dict = meta_optim.state_dict()

        # for layer in range(previous_meta_optim_state_dict['log_init_lr'].shape[0]):
        #     meta_optim_state_dict['log_init_lr'][layer] = previous_meta_optim_state_dict['log_init_lr'][layer]
        #     meta_optim_state_dict['param_group_lstm_hx_init'][:, layer, :] = previous_meta_optim_state_dict['param_group_lstm_hx_init'][:, layer, :]
        #     meta_optim_state_dict['param_group_lstm_cx_init'][:, layer, :] = previous_meta_optim_state_dict['param_group_lstm_cx_init'][:, layer, :]

        # meta_optim_state_dict = {k: previous_meta_optim_state_dict[k]
        #                          if k in previous_meta_optim_state_dict and k not in ['log_init_lr', 'param_group_lstm_hx_init', 'param_group_lstm_cx_init']
        #                          else v
        #                          for k, v in meta_optim_state_dict.items()}

        # meta_optim_state_dict['log_init_lr'] = meta_optim_state_dict['log_init_lr'].expand_as(meta_optim.log_init_lr)
        # # meta_optim_state_dict['log_init_lr'] = meta_optim.log_init_lr
        meta_optim.load_state_dict(previous_meta_optim_state_dict)

    if resume_meta_run_epoch_mode is not None:
        meta_optim.load_state_dict(saved_meta_run['meta_optim_state_dict'])

    _log.info(f"Meta optim model parameters: {sum([p.numel() for p in meta_optim.parameters()])}")

    meta_optim_params = []
    for n, p in meta_optim.named_parameters():
        weight_decay = 0.0
        if 'model_init' in n:
            lr = meta_optim_optim_cfg['model_init_lr']
            weight_decay = meta_optim_optim_cfg['model_init_weight_decay']
        elif 'log_init_lr' in n:
            lr = meta_optim_optim_cfg['log_init_lr_lr']
        elif 'param_group_lstm_hx_init' in n or 'param_group_lstm_cx_init' in n:
            lr = meta_optim_optim_cfg['param_group_lstm_init_lr']
        else:
            lr = meta_optim_optim_cfg['lr']
        meta_optim_params.append({'params': [p], 'lr': lr, 'weight_decay': weight_decay})

    meta_optim_optim = RAdam(meta_optim_params,
    # meta_optim_optim = torch.optim.Adam(meta_optim_params,
                                        lr=meta_optim_optim_cfg['lr'])

    global_rng_state = torch.get_rng_state()

    # model.share_memory()
    meta_optim.share_memory()

    shared_variables = process_manager.dict({'meta_iter': 0, 'meta_epoch': 0})

    shared_meta_optim_grads = {name: torch.zeros_like(param).cpu()
                               for name, param in meta_optim.named_parameters()}
    for grad in shared_meta_optim_grads.values():
        grad.share_memory_()

    if resume_meta_run_epoch_mode is not None:
        shared_variables['meta_iter'] = saved_meta_run['meta_iter']
        shared_variables['meta_epoch'] = saved_meta_run['meta_epoch']

    # start train and val evaluation
    for rank, p in enumerate(eval_processes):
        p['shared_dict'] = process_manager.dict()
        p['shared_dict']['meta_iter'] = None
        p['shared_dict']['best_mean_J'] = 0.0
        rank = rank if gpu_per_dataset_eval else 0
        process_args = [rank, p['dataset_key'], p['flip_label'], meta_optim.state_dict(), shared_variables,
                        _config, p['shared_dict'], save_dir, {n: v.win for n, v in vis_dict.items()},]
        p['process'] = mp.Process(target=evaluate, args=process_args)
        p['process'].start()

    if num_meta_processes:
        meta_mini_batches = list(grouper(meta_batch_size, meta_tasks))
        meta_mini_batch = meta_mini_batches.pop(0)

        sub_meta_batch_size = meta_batch_size // num_meta_processes
        sub_meta_mini_batches = grouper(sub_meta_batch_size, meta_mini_batch)

        for rank, (p, sub_meta_mini_batch) in enumerate(zip(meta_processes, sub_meta_mini_batches)):
            p['shared_dict'] = process_manager.dict()
            p['shared_dict']['sub_iter_done'] = None

            # print('main process',
            #       sub_meta_mini_batch[0]['seq_name'],
            #       sub_meta_mini_batch[0]['train_frame_id'],
            #       sub_meta_mini_batch[0]['meta_frame_id'],
            #       sub_meta_mini_batch[0]['train_transform'].transforms[1].rot)

            p['shared_dict']['sub_meta_mini_batch'] = sub_meta_mini_batch

            process_args = [rank, meta_optim.state_dict(), global_rng_state, _config,
                            datasets['train'], p['shared_dict'], shared_variables,
                            shared_meta_optim_grads]

            p['process'] = mp.Process(target=meta_run, args=process_args)
            p['process'].start()

    start_time = timeit.default_timer()
    meta_epoch_metrics = {'train_loss': {}, 'train_losses': {}, 'meta_loss': {},
                          'meta_losses': {}, 'loss': {}, 'J': {}, 'F': {}}

    while True:
        #
        # VIS EVAL
        #
        for p in eval_processes:
            if p['shared_dict']['meta_iter'] is not None:
                # copy to avoid overwriting by evaluation subprocess
                shared_dict = copy.deepcopy(p['shared_dict'])

                eval_seq_vis = [shared_dict['time_per_frame'],
                                torch.tensor(shared_dict['train_loss_seq']).mean()]

                if _config['parent_model']['architecture'] == 'MaskRCNN':
                    eval_seq_vis.extend([torch.tensor([losses['loss_classifier'] for losses in shared_dict['train_losses_seq']]).mean(),
                                         torch.tensor([losses['loss_box_reg'] for losses in shared_dict['train_losses_seq']]).mean(),
                                         torch.tensor([losses['loss_mask'] for losses in shared_dict['train_losses_seq']]).mean()])

                eval_seq_vis.extend([(torch.tensor(shared_dict['J_seq']).mean() + torch.tensor(shared_dict['F_seq']).mean()) / 2.0,
                                     torch.tensor(shared_dict['J_seq']).mean(),
                                     torch.tensor(shared_dict['J_recall_seq']).mean(),
                                     torch.tensor(shared_dict['F_seq']).mean(),
                                     torch.tensor(shared_dict['F_recall_seq']).mean(),
                                     torch.tensor(shared_dict['init_J_seq']).mean()])
                eval_seq_vis.extend(shared_dict['init_J_seq'])
                eval_seq_vis.extend(shared_dict['J_seq'])
                vis_dict[f"{p['dataset_key']}_flip_label_{p['flip_label']}_eval_seq_vis"].plot(
                    eval_seq_vis, shared_dict['meta_iter'])

                # resume evaluation if not in eval only mode
                if num_meta_processes:
                    p['shared_dict']['meta_iter'] = None

        if num_meta_processes and all([p['shared_dict']['sub_iter_done'] for p in meta_processes]):
            shared_variables['meta_iter'] += 1

            #
            # VIS
            #

            meta_iter_metrics = {'train_loss': [], 'train_losses': [], 'meta_loss': [],
                                 'meta_losses': [], 'loss': [], 'J': [], 'F': []}

            for p in meta_processes:
                shared_dict = p['shared_dict']

                for metric, seqs_values in shared_dict['seqs_metrics'].items():
                    for seq_name, seq_values in seqs_values.items():
                        if seq_name not in meta_epoch_metrics[metric]:
                            meta_epoch_metrics[metric][seq_name] = []

                        if metric == 'meta_loss' and torch.isnan(torch.tensor(seq_values)).any():
                            print(metric, seq_name, shared_variables['meta_iter'])
                            # print('meta_optim.log_init_lr', meta_optim.log_init_lr)

                            print([(t['seq_name'],
                                    t['train_loader'].dataset.multi_object_id,
                                    t['meta_loader'].dataset.multi_object_id,
                                    t['train_loader'].dataset.frame_id,
                                    t['meta_loader'].dataset.frame_id,
                                    t['train_loader'].dataset.augment_with_single_obj_seq_key,
                                    t['meta_loader'].dataset.augment_with_single_obj_seq_key)
                                   for t in shared_dict['sub_meta_mini_batch']])
                            print(list(chain.from_iterable(list(p['shared_dict']['vis_data_seqs'].values()))))
                            exit()

                        meta_epoch_metrics[metric][seq_name].extend(seq_values)
                        meta_iter_metrics[metric].extend(seq_values)
                shared_dict['seqs_metrics'] = {}

            # ITER
            if shared_variables['meta_iter'] == 1 or not shared_variables['meta_iter'] % vis_interval:
                # VIS METRICS
                meta_iter_train_loss = torch.tensor(meta_iter_metrics['train_loss'])
                meta_iter_meta_loss = torch.tensor(meta_iter_metrics['meta_loss'])

                meta_metrics = [meta_iter_train_loss.mean(),
                                meta_iter_meta_loss.mean(),
                                meta_iter_meta_loss.std(),
                                meta_iter_meta_loss.max(),
                                meta_iter_meta_loss.min()]
                meta_metrics.append((timeit.default_timer() - start_time) / 60)
                vis_dict['meta_metrics_vis'].plot(
                    meta_metrics, shared_variables['meta_iter'])

                # VIS LR
                if meta_optim.lr_per_tensor:
                    if _config['num_epochs']['train'] > 1:
                        lrs_hist = []
                        for p in meta_processes:
                            lrs_hist.extend(chain.from_iterable(list(p['shared_dict']['vis_data_seqs'].values())))

                        # [train_loss.item(), bptt_loss.item(), lr.mean().item(), lr.std().item(),]
                        # lrs_hist = torch.Tensor(lrs_hist)

                        vis_dict['lrs_hist_vis'].reset()
                        for epoch in range(_config['num_epochs']['train']):

                            lrs_hist_epoch = [torch.tensor([s[epoch][2] for s in lrs_hist]).mean(),
                                              torch.tensor([s[epoch][2] for s in lrs_hist]).std()]

                            for layer in range(lrs_hist[0][epoch][2].shape[0]):
                                lrs_hist_epoch.append(torch.tensor(
                                    [s[epoch][2][layer] for s in lrs_hist]).mean())

                            vis_dict['lrs_hist_vis'].plot(lrs_hist_epoch, epoch + 1)

                    # log_init_lr = meta_optim.log_init_lr[:, 0]
                    log_init_lr = meta_optim.log_init_lr[:, 0].exp()
                    # log_init_lr = meta_optim.log_init_lr.exp().repeat(1, 2).flatten()
                else:
                    # log_init_lr = torch.Tensor([l.mean() for l in meta_optim.log_init_lr])
                    log_init_lr = torch.Tensor([l.exp().mean() for l in meta_optim.log_init_lr])

                meta_init_lr = [log_init_lr.mean(),
                                log_init_lr.std()]
                meta_init_lr += log_init_lr.detach().numpy().tolist()
                vis_dict['init_lr_vis'].plot(
                    meta_init_lr, shared_variables['meta_iter'])

            # EPOCH
            if not meta_mini_batches:
                shared_variables['meta_epoch'] += 1

                # VIS LOSS
                for loss_name in ['train', 'meta']:
                    meta_loss_seq = [torch.tensor(list(chain.from_iterable(
                                    list(meta_epoch_metrics[f'{loss_name}_loss'].values())))).mean()]

                    if _config['parent_model']['architecture'] == 'MaskRCNN':
                        # 'MEAN cls_score', 'MEAN bbox_pred', 'MEAN mask_fcn_logits'
                        meta_losses_list = list(chain.from_iterable(list(meta_epoch_metrics[f'{loss_name}_losses'].values())))
                        meta_loss_seq.append(torch.tensor(
                            [v['loss_classifier'] for v in meta_losses_list]).mean())
                        meta_loss_seq.append(torch.tensor(
                            [v['loss_box_reg'] for v in meta_losses_list]).mean())
                        meta_loss_seq.append(torch.tensor(
                            [v['loss_mask'] for v in meta_losses_list]).mean())

                    for m_l in meta_epoch_metrics[f'{loss_name}_loss'].values():
                        meta_loss_seq.append(torch.tensor(m_l).mean())
                    vis_dict[f'{loss_name}_loss_seq_vis'].plot(
                        meta_loss_seq, shared_variables['meta_epoch'])

                # SAVE MODEL
                if save_dir is not None:
                    save_meta_run = {'meta_optim_state_dict': meta_optim.state_dict(),
                                    #  'meta_optim_optim_state_dict': meta_optim_optim.state_dict(),
                                     'vis_win_names': {n: v.win for n, v in vis_dict.items()},
                                     'meta_iter': shared_variables['meta_iter'],
                                     'meta_epoch': shared_variables['meta_epoch']}
                    torch.save(save_meta_run, os.path.join(
                        save_dir, f"last_meta_epoch.model"))

                meta_epoch_metrics = {'train_loss': {}, 'train_losses': {}, 'meta_loss': {},
                                      'meta_losses': {}, 'loss': {}, 'J': {}, 'F': {}}

                meta_tasks = generate_meta_train_tasks()  # pylint: disable=E1120
                meta_mini_batches = list(grouper(meta_batch_size, meta_tasks))

            #
            # STEP
            #
            # normalize over batch and clip grads
            for name, param in meta_optim.named_parameters():
                param.grad = shared_meta_optim_grads[name] / meta_batch_size

                grad_clip = _config['meta_optim_optim_cfg']['grad_clip']
                if grad_clip is not None:
                    param.grad.clamp_(-1.0 * grad_clip, grad_clip)

            meta_optim_optim.step()
            meta_optim_optim.zero_grad()
            for grad in shared_meta_optim_grads.values():
                grad.zero_()

            if meta_optim.lr_per_tensor:
                meta_optim.log_init_lr.data.clamp_(-33, 33)
            else:
                meta_optim.log_init_lr = [l.data.clamp_(-33, 33)
                                          for l in meta_optim.log_init_lr]

            # global_rng_state = process['shared_dict']['global_rng_state']
            # reset iter_done and continue with next iter in meta_run subprocess
            meta_mini_batch = meta_mini_batches.pop(0)
            sub_meta_mini_batches = grouper(sub_meta_batch_size, meta_mini_batch)

            for p, sub_meta_mini_batch in zip(meta_processes, sub_meta_mini_batches):
                p['shared_dict']['sub_meta_mini_batch'] = sub_meta_mini_batch
                p['shared_dict']['sub_iter_done'] = False
            start_time = timeit.default_timer()

            # for p in meta_processes:
            #     for seq_name, vis_data in p['shared_dict']['vis_data_seqs'].items():
            #         # Visdom throws no error for infinte or NaN values
            #         assert not np.isnan(np.array(vis_data)).any() and \
            #             np.isfinite(np.array(vis_data)).all()

            #         vis_dict[f"{seq_name}_model_metrics"].reset()
            #         for epoch, vis_datum in enumerate(vis_data):
            #             vis_dict[f"{seq_name}_model_metrics"].plot(vis_datum, epoch + 1)
