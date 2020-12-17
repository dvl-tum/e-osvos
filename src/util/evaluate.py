import copy
import logging
import os
import time
import timeit

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from data import custom_transforms
from meta_optim.meta_optim import MetaOptimizer
from networks.mask_rcnn import MaskRCNN

from util.helper_func import (compute_loss, data_loaders, early_stopping,
                              epoch_iter, eval_davis_seq, eval_loader,
                              init_parent_model, run_loader, set_random_seeds)


def evaluate(rank: int, dataset_key: str,
             shared_meta_optim_state_dict: dict, shared_variables: dict,
             _config: dict, shared_dict: dict, save_dir: str,
             vis_win_names: dict, evaluate_only: bool, _log: logging):
    seed = _config['seed']
    loss_func = _config['loss_func']
    datasets = _config['datasets']

    torch.backends.cudnn.fastest = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data_cfg = copy.deepcopy(_config['data_cfg'])

    while True:
        while shared_dict['meta_iter'] is not None:
            time.sleep(0.25)

        meta_optim_state_dict = copy.deepcopy(shared_meta_optim_state_dict)
        meta_iter = shared_variables['meta_iter']
        meta_epoch = shared_variables['meta_epoch']

        set_random_seeds(seed)

        device = torch.device(f'cuda:{rank}')

        model, parent_states = init_parent_model(**_config['parent_model'])
        if dataset_key in parent_states and parent_states[dataset_key]['states']:
            if len(parent_states[dataset_key]['states']) > 1:
                raise NotImplementedError
            model.load_state_dict(parent_states[dataset_key]['states'][0])

        meta_optim = MetaOptimizer(model, **_config['meta_optim_cfg'])
        meta_optim.load_state_dict(meta_optim_state_dict)

        model.to(device)
        meta_optim.to(device)

        train_loader, test_loader, meta_loader = data_loaders(  # pylint: disable=E1120
            datasets[dataset_key], **data_cfg)
        # remove random cropping
        train_loader.dataset.crop_size = None
        test_loader.dataset.crop_size = None
        meta_loader.dataset.crop_size = None

        def early_stopping_func(loss_hist):
            return early_stopping(loss_hist, **_config['train_early_stopping_cfg'])

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

            preds_save_dir = os.path.join(save_dir,
                                          'best_eval_preds',
                                          f"{datasets[dataset_key]['name']}",
                                          f"{datasets[dataset_key]['split']}")
            if not os.path.exists(preds_save_dir):
                os.makedirs(preds_save_dir)
            for seq_name in train_loader.dataset.seqs_names:
                if not os.path.exists(os.path.join(preds_save_dir, seq_name)):
                    os.makedirs(os.path.join(preds_save_dir, seq_name))

        eval_time = 0
        num_frames = 0
        init_J_seq = []
        J_seq = []
        J_recall_seq = []
        J_decay_seq = []
        train_loss_seq = []
        train_losses_seq = []
        F_seq = []
        F_recall_seq = []
        F_decay_seq = []
        masks = {}
        boxes = {}

        if _config['data_cfg']['multi_object'] == 'single_id':
            model.roi_heads.detections_per_img = 1

        random_transformation_transforms = train_loader.dataset.transform

        for seq_name in train_loader.dataset.seqs_names:
            train_loader.dataset.set_seq(seq_name)
            test_loader.dataset.set_seq(seq_name)
            meta_loader.dataset.set_seq(seq_name)

            if train_loader.dataset.num_object_groups == 1:
                test_loader.dataset.multi_object_id = 0

                meta_optim.load_state_dict(meta_optim_state_dict)
                meta_optim.reset()
                meta_optim.eval()

                if test_loader.dataset.test_mode or test_loader.dataset.all_frames:
                    J = [0.0]
                else:
                    _, _, J, _,  = eval_loader(model, test_loader, loss_func)
                init_J_seq.extend(J)

            boxes[seq_name] = [None] * len(test_loader.dataset)
            masks[seq_name] = []

            for obj_id in range(train_loader.dataset.num_object_groups):
                train_loader.dataset.multi_object_id = obj_id
                test_loader.dataset.multi_object_id = obj_id
                meta_loader.dataset.multi_object_id = obj_id

                train_loader.dataset.set_gt_frame_id()
                num_objects_in_group = train_loader.dataset.num_objects_in_group

                # evaluation with online adaptation
                if _config['eval_online_adapt']['step']:
                    eval_online_adapt_step = _config['eval_online_adapt']['step']
                    meta_frame_iter = range(train_loader.dataset.frame_id + 1,
                                            len(test_loader.dataset),
                                            eval_online_adapt_step)
                else:
                    # one iteration with original meta frame and evaluation of entire sequence
                    eval_online_adapt_step = len(test_loader.dataset)
                    meta_frame_iter = [meta_loader.dataset.frame_id]

                # meta_frame_id might be a str, e.g., 'middle'
                start_eval = timeit.default_timer()
                for eval_online_step_count, _ in enumerate(meta_frame_iter):
                    # range [min, max[
                    if eval_online_step_count == 0:
                        train_frame = test_loader.dataset[train_loader.dataset.frame_id]
                        train_frame_gt = train_frame['gt']

                        for frame_id in range(len(test_loader.dataset)):
                            if not obj_id:
                                masks[seq_name].append(torch.zeros(num_objects_in_group, train_frame_gt.shape[1], train_frame_gt.shape[2]))
                            else:
                                masks[seq_name][frame_id] = torch.cat([
                                    masks[seq_name][frame_id],
                                    torch.zeros(num_objects_in_group, train_frame_gt.shape[1], train_frame_gt.shape[2])])

                        masks[seq_name][train_loader.dataset.frame_id][obj_id,
                                                                       :, :] = 2 * train_frame_gt

                        eval_frame_range_min = train_loader.dataset.frame_id + 1
                        eval_frame_range_max = eval_frame_range_min # + eval_online_adapt_step // 2
                    else:
                        # eval_frame_range_min = (meta_frame_id - eval_online_adapt_step // 2) + 1
                        eval_frame_range_min = eval_frame_range_max

                        propagate_frame_gt = masks[seq_name][eval_frame_range_min -
                                                             1][obj_id: obj_id + 1].ge(_config['eval_online_adapt']['min_prop']).float()

                        propagate_frame_gts = []
                        for propagate_frame_id in range(1, _config['eval_online_adapt']['step']):

                            propagate_frame_gt_numpy = masks[seq_name][eval_frame_range_min -
                                                                       propagate_frame_id][obj_id: obj_id + 1].ge(_config['eval_online_adapt']['min_prop']).float()

                            propagate_frame_gt_numpy = np.copy(np.transpose(propagate_frame_gt_numpy.cpu().numpy(), (1, 2, 0)))

                            propagate_frame_gts.append(
                                propagate_frame_gt_numpy)

                    eval_frame_range_max += eval_online_adapt_step
                    # if eval_frame_range_max + (eval_online_adapt_step // 2 + 1) > len(test_loader.dataset):
                    if eval_frame_range_max > len(test_loader.dataset):
                        eval_frame_range_max = len(test_loader.dataset)

                    # load_state_dict(model, seq_name, parent_states[dataset_key])
                    if eval_online_step_count == 0 or _config['eval_online_adapt']['reset_model_mode'] == 'FULL':
                        meta_optim.load_state_dict(meta_optim_state_dict)
                        meta_optim.reset()
                        meta_optim.eval()
                    elif _config['eval_online_adapt']['reset_model_mode'] == 'FIRST_STEP':
                        meta_optim.load_state_dict(meta_optim_state_dict)

                        model.load_state_dict(model_state_dict_first_step)

                        meta_optim.eval()

                    train_loss_hist = []
                    if eval_online_step_count == 0:
                        num_epochs = _config['num_epochs']['eval']
                    else:
                        num_epochs = _config['eval_online_adapt']['num_epochs']

                    model.train_without_dropout()

                    if eval_online_step_count:
                        train_loader.dataset.transform = custom_transforms.ToTensor()
                    else:
                        train_loader.dataset.transform = random_transformation_transforms

                    for epoch in epoch_iter(num_epochs):
                        set_random_seeds(
                            _config['seed'] + epoch + eval_online_step_count)

                        for _, sample_batched in enumerate(train_loader):
                            inputs, gts = sample_batched['image'], sample_batched['gt']

                            if eval_online_step_count:
                                inputs = inputs[:1]
                                gts = gts[:1]

                                num_propagte_frames = min(
                                    _config['eval_online_adapt']['step'],
                                    _config['data_cfg']['batch_sizes']['train'])
                                start_propagate_frame = _config['eval_online_adapt']['step'] - num_propagte_frames + 1

                                for propagate_frame_id in range(start_propagate_frame, _config['eval_online_adapt']['step']):
                                    propagate_frame_gt_numpy = propagate_frame_gts[propagate_frame_id - 1]

                                    if (propagate_frame_gt_numpy == 1.0).astype(float).sum().item() != 0:
                                        train_loader.dataset.frame_id = eval_frame_range_min - propagate_frame_id
                                        train_loader.dataset.propagate_frame_gt = propagate_frame_gt_numpy

                                        for _, sample_batched in enumerate(train_loader):
                                            inputs_propagate, gts_propagate = sample_batched['image'], sample_batched['gt']

                                        inputs = torch.cat(
                                            [inputs,
                                            inputs_propagate[:1]])
                                        gts = torch.cat([gts,
                                                        gts_propagate[:1]])

                                train_loader.dataset.propagate_frame_gt = None
                                train_loader.dataset.set_gt_frame_id()

                            inputs, gts = inputs.to(device), gts.to(device)

                            if isinstance(model, MaskRCNN):
                                train_loss, train_losses = model(inputs, gts)
                            else:
                                outputs = model(inputs)
                                train_loss = compute_loss(loss_func, outputs[-1], gts)

                            train_loss_hist.append(train_loss.item())

                            model.zero_grad()

                            meta_optim.set_train_loss(train_loss)

                            if _config['eval_online_adapt']['reset_model_mode'] == 'FIRST_STEP':
                                meta_optim.only_box_head = eval_online_step_count != 0

                            meta_optim.step(train_loss)

                            meta_optim.meta_model.detach_param_groups()

                            if early_stopping_func(train_loss_hist):
                                break

                        if early_stopping_func(train_loss_hist):
                            break
                    train_loss_seq.append(train_loss.item())

                    if eval_online_step_count == 0:
                        # meta_optim_state_dict_first_step = copy.deepcopy(
                        #     meta_optim.state_dict())
                        model_state_dict_first_step = copy.deepcopy(
                            model.state_dict())

                    if _config['parent_model']['architecture'] == 'MaskRCNN':
                        train_losses_seq.append({k: v.cpu().item()
                                                for k, v in train_losses.items()})

                    # run model on frame range
                    test_loader.sampler.indices = range(eval_frame_range_min, eval_frame_range_max)

                    if eval_online_step_count == 0:
                        targets = train_frame_gt.unsqueeze(dim=0)
                    else:
                        targets = propagate_frame_gt.unsqueeze(dim=0)

                    _, _, probs_frame_range, boxes_frame_range = run_loader(model, test_loader, loss_func, return_probs=True, start_targets=targets)
                    probs_frame_range = probs_frame_range.cpu()
                    boxes_frame_range = boxes_frame_range.cpu()

                    for frame_id, probs, box in zip(test_loader.sampler.indices, probs_frame_range, boxes_frame_range):

                        if boxes[seq_name][frame_id] is None:
                            boxes[seq_name][frame_id] = box
                        else:
                            boxes[seq_name][frame_id] = torch.cat([boxes[seq_name][frame_id], box])

                        masks[seq_name][frame_id][-num_objects_in_group:, :, :] = probs

                    test_loader.sampler.indices = None

                    if eval_frame_range_max == len(test_loader.dataset):
                        break

                eval_time += (timeit.default_timer() - start_eval)
                num_frames += len(test_loader.dataset)

            # merge all logit maps and set object predictions by argmax
            for frame_id in range(len(test_loader.dataset)):
                background_mask = masks[seq_name][frame_id].max(dim=0, keepdim=True)[0].lt(0.5)
                masks[seq_name][frame_id] = masks[seq_name][frame_id].argmax(dim=0, keepdim=True).float() + 1.0
                masks[seq_name][frame_id][background_mask] = 0.0

            # TODO: refactor
            # assert test_loader.dataset.frame_id is None
            test_loader_frame_id = test_loader.dataset.frame_id
            test_loader.dataset.frame_id = None
            for frame_id, mask_frame in enumerate(masks[seq_name]):
                file_name = test_loader.dataset[frame_id]['file_name']

                if test_loader.dataset.all_frames and not any([file_name in l for l in test_loader.dataset.labels]):
                    continue

                mask_frame = np.transpose(mask_frame.cpu().numpy(), (1, 2, 0)).astype(np.uint8)

                pred_path = os.path.join(preds_save_dir, seq_name, os.path.basename(file_name) + '.png')

                imageio.imsave(pred_path, mask_frame)
            test_loader.dataset.frame_id = test_loader_frame_id

            if test_loader.dataset.test_mode:
                evaluation = {'J': {'mean': [0.0], 'recall': [0.0], 'decay': [0.0]},
                              'F': {'mean': [0.0], 'recall': [0.0], 'decay': [0.0]}}
            else:
                evaluation = eval_davis_seq(preds_save_dir, seq_name)

            if evaluate_only:
                _log.info(f"{dataset_key}: {seq_name} {evaluation['J']['mean']}")

            J_seq.extend(evaluation['J']['mean'])
            J_recall_seq.extend(evaluation['J']['recall'])
            J_decay_seq.extend(evaluation['J']['decay'])
            F_seq.extend(evaluation['F']['mean'])
            F_recall_seq.extend(evaluation['F']['recall'])
            F_decay_seq.extend(evaluation['F']['decay'])

        if save_dir is not None:
            if not test_loader.dataset.test_mode:
                save_meta_run = {'meta_optim_state_dict': meta_optim.state_dict(),
                                'vis_win_names': vis_win_names,
                                'meta_iter': meta_iter,
                                'meta_epoch': meta_epoch,}
                torch.save(save_meta_run, os.path.join(
                    save_dir, f"last_{dataset_key}_meta_iter.model"))

        mean_J = torch.tensor(J_seq).mean().item()

        if test_loader.dataset.test_mode or mean_J > shared_dict['best_mean_J']:
            shared_dict['best_mean_J'] = mean_J

            if save_dir is not None:
                if not test_loader.dataset.test_mode:
                    save_meta_run = {'meta_optim_state_dict': meta_optim.state_dict(),
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

                        pred_path = os.path.join(
                            debug_preds_save_dir,
                            seq_name,
                            os.path.basename(file_name) + '.png')

                        # TODO: implement color palette for labels
                        fig = plt.figure()
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)

                        ax.imshow(mask_frame.squeeze(2), cmap='jet', vmin=0, vmax=test_loader.dataset.num_objects)

                        if frame_id and boxes_seq[frame_id] is not None:
                            for obj_id, box in enumerate(boxes_seq[frame_id]):
                                ax.add_patch(
                                    plt.Rectangle(
                                        (box[0], box[1]),
                                        box[2] - box[0],
                                        box[3] - box[1],
                                        fill=False,
                                        linewidth=1.0,
                                    ))

                                ax.annotate(obj_id, (box[0] + (box[2] - box[0]) / 2.0, box[1] + (box[3] - box[1]) / 2.0),
                                            weight='bold', fontsize=12, ha='center', va='center')

                        plt.axis('off')
                        plt.draw()
                        plt.savefig(pred_path, dpi=100)
                        plt.close()
                test_loader.dataset.frame_id = test_loader_frame_id

        shared_dict['init_J_seq'] = init_J_seq
        shared_dict['J_seq'] = J_seq
        shared_dict['J_recall_seq'] = J_recall_seq
        shared_dict['J_decay_seq'] = J_decay_seq
        shared_dict['train_losses_seq'] = train_losses_seq
        shared_dict['train_loss_seq'] = train_loss_seq
        shared_dict['F_seq'] = F_seq
        shared_dict['F_recall_seq'] = F_recall_seq
        shared_dict['F_decay_seq'] = F_decay_seq
        shared_dict['time_per_frame'] = eval_time / num_frames

        # set meta_iter here to signal main process that eval is finished
        shared_dict['meta_iter'] = meta_iter
