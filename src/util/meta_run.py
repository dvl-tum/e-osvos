import time

import torch
from meta_optim.meta_tasksets import MetaTaskset
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from meta_optim.meta_optim import MetaOptimizer
from .helper_func import (compute_loss, data_loaders, device_for_process,
                          early_stopping, epoch_iter, grouper,
                          init_parent_model, load_state_dict, train_val,
                          set_random_seeds)


def meta_run(rank: int, init_model_state_dict: dict,
             shared_meta_optim_state_dict: dict,
             global_rng_state: torch.ByteTensor, _config: dict, dataset: str,
             shared_dict: dict, shared_variables: dict,
             shared_meta_optim_grads: dict, save_dir: str,
             num_meta_processes: int):

    device, meta_device = device_for_process(rank,
                                             _config['eval_datasets'],
                                             _config['num_meta_processes_per_gpu'],
                                             _config['num_eval_gpus'])

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_random_seeds(_config['seed'] + rank)

    model, _ = init_parent_model(**_config['parent_model'])
    model.load_state_dict(init_model_state_dict)

    meta_optim = MetaOptimizer(model, **_config['meta_optim_cfg'])

    num_epochs = _config['num_epochs']['train']

    sub_meta_batch_size = _config['meta_batch_size'] // num_meta_processes

    meta_task_set_config = (
        _config['random_frame_transform_per_task'],
        _config['random_flip_label'],
        _config['random_no_label'],
        _config['data_cfg'],
        _config['single_obj_seq_mode'],
        _config['random_box_coord_perm'],
        _config['random_frame_epsilon'],
        _config['random_object_id_sub_group'])

    if isinstance(_config['datasets']['train']['name'], list):
        meta_task_sets = []
        for n, s in zip(_config['datasets']['train']['name'],
                        _config['datasets']['train']['split']):

            train_loader, test_loader, meta_loader = data_loaders(
                {'name': n, 'split': s}, **_config['data_cfg'])

            meta_task_set = MetaTaskset(
                train_loader, test_loader, meta_loader,
                *meta_task_set_config)

            meta_task_sets.append(meta_task_set)

        meta_task_set = ConcatDataset(meta_task_sets)
    else:
        train_loader, test_loader, meta_loader = data_loaders(
                _config['datasets']['train'], **_config['data_cfg'])

        meta_task_set = MetaTaskset(
            train_loader, test_loader, meta_loader, *meta_task_set_config)

    def collate_fn(batch):
        return batch

    meta_task_loader = DataLoader(
        meta_task_set,
        shuffle=True,
        batch_size=sub_meta_batch_size,
        num_workers=0,
        collate_fn=collate_fn)

    while True:

        for meta_mini_batch in meta_task_loader:

            # main process sets iter_done=False after shared_meta_optim is updated
            while shared_dict['sub_iter_done']:
                time.sleep(0.25)

            # filter None values from grouper
            meta_mini_batch = [s for s in meta_mini_batch if s is not None]

            # model.load_state_dict(model_state_dict)
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

                bptt_loss = torch.zeros(1).to(meta_device)
                stop_train = False
                prev_bptt_iter_loss = torch.zeros(1).to(meta_device)
                train_loss_hist = []
                train_losses_hist = []
                vis_data_seqs_sample = []

                meta_optim.reset()
                meta_optim.zero_grad()

                for epoch in epoch_iter(num_epochs):
                    if _config['increase_seed_per_meta_run']:
                        set_random_seeds(_config['seed'] + rank + epoch + shared_variables['meta_iter'])
                    else:
                        set_random_seeds(_config['seed'] + rank + epoch)

                    model.train_without_dropout()

                    # only single iteration
                    for train_batch in train_loader:
                        train_inputs, train_gts = train_batch['image'], train_batch['gt']
                        train_inputs, train_gts = train_inputs.to(device), train_gts.to(device)

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

                    meta_optim.set_train_loss(train_loss)
                    meta_optim.step(train_loss)

                    if _config['multi_step_bptt_loss']:
                        assert num_epochs == len(_config['multi_step_bptt_loss'])

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

                        bptt_loss += _config['multi_step_bptt_loss'][epoch - 1] * bptt_iter_loss  # - prev_bptt_iter_loss

                    # visualization

                    vis_data = [train_loss.item(),
                                bptt_loss.item(),
                                meta_optim.state_lr.cpu().detach().numpy()]
                    vis_data_seqs_sample.append(vis_data)

                    stop_train = early_stopping(
                        train_loss_hist, **_config['train_early_stopping_cfg']) or epoch == num_epochs

                    # update params of meta optim
                    if not epoch % _config['bptt_epochs'] or stop_train:

                        if not _config['multi_step_bptt_loss']:
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

                                bptt_loss += meta_loss

                        bptt_loss_is_nan = torch.isnan(bptt_loss).any()
                        if bptt_loss_is_nan:
                            stop_train = True

                        # meta_optim.zero_grad()
                        bptt_loss.backward()

                        if not stop_train:
                            meta_optim.reset(keep_state=True)

                            prev_bptt_iter_loss.zero_().detach_()
                            bptt_loss.zero_().detach_()

                    if stop_train:
                        meta_optim.reset()
                        break

                if not bptt_loss_is_nan:
                    seqs_metrics['meta_loss'][seq_name].append(meta_loss.item())
                    seqs_metrics['meta_losses'][seq_name].append({k: v.cpu().item()
                                                                for k, v in meta_losses.items()})

                    vis_data_seqs[seq_name].append(vis_data_seqs_sample)

                    seqs_metrics['train_loss'][seq_name].append(train_loss_hist[0])
                    if _config['parent_model']['architecture'] == 'MaskRCNN':
                        seqs_metrics['train_losses'][seq_name].append(train_losses_hist[0])

                    for name, param in meta_optim.named_parameters():
                        shared_meta_optim_grads[name] += param.grad.cpu()

            shared_dict['seqs_metrics'] = seqs_metrics
            shared_dict['vis_data_seqs'] = vis_data_seqs
            shared_dict['global_rng_state'] = global_rng_state
            shared_dict['sub_iter_done'] = True
