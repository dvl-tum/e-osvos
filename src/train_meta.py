import copy
import logging
import os
import shutil
import timeit
from itertools import chain

import numpy as np
import sacred
import torch
import torch.multiprocessing as mp
import torchvision

from meta_optim.meta_optim import MetaOptimizer
from util.helper_func import (init_parent_model, load_state_dict, set_random_seeds)
from util.radam import RAdam
from util.visualize import init_vis
from util.meta_run import meta_run
from util.evaluate import evaluate

ex = sacred.Experiment('e-osvos-meta')
ex.add_config('cfgs/meta.yaml')
ex.add_config('cfgs/torch.yaml')
ex.add_named_config('DAVIS-2017', 'cfgs/meta_davis-2017.yaml')
ex.add_named_config('YouTube-VOS', 'cfgs/meta_youtube-vos.yaml')
ex.add_named_config('e-OSVOS', 'cfgs/eval_e-osvos.yaml')
ex.add_named_config('e-OSVOS-OnA', 'cfgs/eval_e-osvos-OnA.yaml')


MetaOptimizer = ex.capture(MetaOptimizer, prefix='meta_optim_cfg')
init_vis = ex.capture(init_vis)


@ex.capture
def get_run_name(datasets, env_suffix):
    dataset_name = str(datasets['train']['name']).replace(', ', '+').replace("'", '').replace('[', '').replace(']', '')
    if env_suffix is None:
        return dataset_name
    return f"{dataset_name}_{env_suffix}"

@ex.automain
def main(save_dir: str, resume_meta_run_epoch_mode: str, env_suffix: str,
         eval_datasets: bool, num_meta_processes_per_gpu: int, datasets: dict,
         meta_optim_optim_cfg: dict, seed: int, _config: dict, _log: logging,
         meta_optim_model_file: str, num_eval_gpus: int, no_vis: bool,
         meta_batch_size: int, vis_interval: int, data_cfg: dict, torch_cfg: dict,
         _run: sacred.run.Run):
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')

    if torch_cfg['print_config'] and _log.getEffectiveLevel() < 30:
        sacred.commands.print_config(_run)

    assert datasets['train'] is not None

    set_random_seeds(seed)

    vis_dict = init_vis(get_run_name())  # pylint: disable=E1120

    assert save_dir is not None

    save_dir = os.path.join(save_dir, get_run_name())  # pylint: disable=E1120
    if os.path.exists(save_dir):
        if resume_meta_run_epoch_mode is None:
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)

    if resume_meta_run_epoch_mode is not None:
        if resume_meta_run_epoch_mode == 'LAST':
            resume_model_name = f"last_meta_iter.model"
        elif 'BEST' in resume_meta_run_epoch_mode:
            resume_model_name = f"best_{resume_meta_run_epoch_mode.split('_')[1].lower()}_meta_iter.model"
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
    # Meta model
    #
    model, parent_states = init_parent_model(**_config['parent_model'])

    if 'train' in parent_states and parent_states['train']['states']:
        if len(parent_states['train']['states']) > 1:
            raise NotImplementedError
        model.load_state_dict(parent_states['train']['states'][0])

    meta_optim = MetaOptimizer(model)  # pylint: disable=E1120
    meta_optim.init_zero_grad()

    if meta_optim_model_file is not None:
        previous_meta_optim_state_dict = torch.load(meta_optim_model_file)['meta_optim_state_dict']
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
        else:
            lr = meta_optim_optim_cfg['lr']

        if meta_optim_optim_cfg['freeze_encoder'] and ('backbone' in n or 'rpn' in n):
            lr = 0.0

        meta_optim_params.append({'params': [p], 'lr': lr, 'weight_decay': weight_decay})

    meta_optim_optim = RAdam(
        meta_optim_params, lr=meta_optim_optim_cfg['lr'])

    #
    # processes
    #
    num_meta_processes = torch.cuda.device_count()
    eval_processes = {}
    if eval_datasets:
        eval_processes = [{'dataset_key': k}
                          for k, v in datasets.items()
                          if v['eval'] and v['split'] is not None]

        if num_eval_gpus is None:
            num_eval_gpus = len(eval_processes)
            _config['num_eval_gpus'] = num_eval_gpus
        assert len(eval_processes) % num_eval_gpus == 0

        num_meta_processes -= num_eval_gpus

        assert num_meta_processes >= 0

    if num_meta_processes and num_meta_processes_per_gpu:
        num_meta_processes *= num_meta_processes_per_gpu
        assert not meta_batch_size % num_meta_processes, ('meta_batch_size is not a multiple of num_meta_processes.')
    else:
        num_meta_processes = 0
        _log.warning(f"EVAL modus.")

    process_manager = mp.Manager()
    meta_processes = [dict() for _ in range(num_meta_processes)]

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

        rank = rank % num_eval_gpus

        process_args = [rank, p['dataset_key'], meta_optim.state_dict(), shared_variables,
                        _config, p['shared_dict'], save_dir, {n: v.win for n, v in vis_dict.items()},
                        not bool(num_meta_processes), _log]
        p['process'] = mp.Process(target=evaluate, args=process_args)
        p['process'].start()

    if num_meta_processes:
        # for rank, (p, sub_meta_mini_batch) in enumerate(zip(meta_processes, sub_meta_mini_batches)):
        for rank, p in enumerate(meta_processes):
            p['shared_dict'] = process_manager.dict()
            p['shared_dict']['sub_iter_done'] = False
            p['shared_dict']['meta_epoch_done'] = False

            process_args = [rank, model.state_dict(), meta_optim.state_dict(),
                            global_rng_state, _config, datasets['train'],
                            p['shared_dict'], shared_variables, shared_meta_optim_grads,
                            save_dir, num_meta_processes]

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
                    eval_seq_vis.extend([
                        torch.tensor([losses['loss_classifier'] for losses in shared_dict['train_losses_seq']]).mean(),
                        torch.tensor([losses['loss_box_reg'] for losses in shared_dict['train_losses_seq']]).mean(),
                        torch.tensor([losses['loss_mask'] for losses in shared_dict['train_losses_seq']]).mean()])

                eval_seq_vis.extend([(
                    torch.tensor(shared_dict['J_seq']).mean() + torch.tensor(shared_dict['F_seq']).mean()) / 2.0,
                    torch.tensor(shared_dict['J_seq']).mean(),
                    torch.tensor(shared_dict['J_recall_seq']).mean(),
                    torch.tensor(shared_dict['J_decay_seq']).mean(),
                    torch.tensor(shared_dict['F_seq']).mean(),
                    torch.tensor(shared_dict['F_recall_seq']).mean(),
                    torch.tensor(shared_dict['F_decay_seq']).mean(),
                    torch.tensor(shared_dict['init_J_seq']).mean()])
                eval_seq_vis.extend(shared_dict['init_J_seq'])
                eval_seq_vis.extend(shared_dict['J_seq'])

                if not no_vis:
                    vis_dict[f"{p['dataset_key']}_eval_seq_vis"].plot(
                        eval_seq_vis, shared_dict['meta_iter'])

                _log.info(f"{p['dataset_key']}: J mean {torch.tensor(shared_dict['J_seq']).mean():.1%}")

                # evalutate only once if in eval mode
                if not num_meta_processes:
                    p['process'].terminate()
                    p['shared_dict']['meta_iter'] = None
                else:
                    p['shared_dict']['meta_iter'] = None

        # finish in eval mode when all evaluations are done
        if not num_meta_processes and all([not p['process'].is_alive() for p in eval_processes]):
            return

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

                        meta_epoch_metrics[metric][seq_name].extend(seq_values)
                        meta_iter_metrics[metric].extend(seq_values)
                shared_dict['seqs_metrics'] = {}

            # ITER
            if shared_variables['meta_iter'] == 1 or not shared_variables['meta_iter'] % vis_interval:
                # SAVE MODEL
                if save_dir is not None:
                    save_meta_run = {'meta_optim_state_dict': meta_optim.state_dict(),
                                    #  'meta_optim_optim_state_dict': meta_optim_optim.state_dict(),
                                     'vis_win_names': {n: v.win for n, v in vis_dict.items()},
                                     'meta_iter': shared_variables['meta_iter'],
                                     'meta_epoch': shared_variables['meta_epoch']}
                    torch.save(save_meta_run, os.path.join(
                        save_dir, f"last_meta_iter.model"))

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
                if _config['num_epochs']['train'] > 1:
                    lrs_hist = []
                    for p in meta_processes:
                        lrs_hist.extend(chain.from_iterable(list(p['shared_dict']['vis_data_seqs'].values())))

                    vis_dict['lrs_hist_vis'].reset()
                    for epoch in range(_config['num_epochs']['train']):

                        lrs_hist_epoch = [torch.tensor([s[epoch][2] for s in lrs_hist]).mean(),
                                          torch.tensor([s[epoch][2] for s in lrs_hist]).std()]

                        for layer in range(lrs_hist[0][epoch][2].shape[0]):
                            lrs_hist_epoch.append(torch.tensor(
                                [s[epoch][2][layer] for s in lrs_hist]).mean())

                        vis_dict['lrs_hist_vis'].plot(lrs_hist_epoch, epoch + 1)

                meta_init_lr = [meta_optim.init_lr.mean(),
                                meta_optim.init_lr.std()]
                meta_init_lr += meta_optim.init_lr.detach().numpy().tolist()
                vis_dict['init_lr_vis'].plot(
                    meta_init_lr, shared_variables['meta_iter'])

            # EPOCH
            if all([p['shared_dict']['meta_epoch_done'] for p in meta_processes]):
            # if not meta_mini_batches:
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

                    vis_dict[f'{loss_name}_loss_seq_vis'].plot(
                        meta_loss_seq, shared_variables['meta_epoch'])

                meta_epoch_metrics = {'train_loss': {}, 'train_losses': {}, 'meta_loss': {},
                                      'meta_losses': {}, 'loss': {}, 'J': {}, 'F': {}}

                for p in meta_processes:
                    p['shared_dict']['meta_epoch_done'] = False

            #
            # STEP
            #
            # normalize over batch and clip grads

            start_time = timeit.default_timer()

            for name, param in meta_optim.named_parameters():
                param.grad = shared_meta_optim_grads[name] / meta_batch_size

                grad_clip = _config['meta_optim_optim_cfg']['grad_clip']
                if grad_clip is not None:
                    param.grad.clamp_(-1.0 * grad_clip, grad_clip)

            meta_optim_optim.step()
            meta_optim_optim.zero_grad()
            for grad in shared_meta_optim_grads.values():
                grad.zero_()

            meta_optim.clamp_init_lr()

            for p in meta_processes:
                # p['shared_dict']['sub_meta_mini_batch'] = sub_meta_mini_batch
                p['shared_dict']['sub_iter_done'] = False
