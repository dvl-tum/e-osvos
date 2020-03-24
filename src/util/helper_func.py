import collections
import os
import shutil
import tempfile
from itertools import count, product, zip_longest

import davis
import imageio
import numpy as np
import torch
import torch.nn as nn
from data import DAVIS, YouTube, custom_transforms
from davis import (Annotation, DAVISLoader, Segmentation, db_eval,
                   db_eval_sequence)
from layers.osvos_layers import class_balanced_cross_entropy_loss, dice_loss
from meta_optim.meta_optim import MetaOptimizer
from networks.drn_seg import DRNSeg
from networks.fpn import FPN
from networks.unet import Unet
from networks.vgg_osvos import OSVOSVgg
from networks.deeplabv3 import DeepLabV3
from networks.deeplabv3plus import DeepLabV3Plus
from networks.mask_rcnn import MaskRCNN
from prettytable import PrettyTable
from pytorch_tools.data import EpochSampler
from pytorch_tools.ingredients import set_random_seeds
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms


def compute_loss(loss_func, outputs, gts, loss_kwargs=None):
    if loss_kwargs is None:
        loss_kwargs = {}

    if loss_func == 'cross_entropy':
        reduction = 'mean'
        if 'batch_average' in loss_kwargs and not loss_kwargs['batch_average']:
            reduction = 'none'
        criterion = nn.BCEWithLogitsLoss(reduction=reduction).cuda()
        loss = criterion(outputs, gts)
        if reduction == 'none':
            loss = loss.view(loss.shape[0], -1).mean(dim=1)
        return loss
    elif loss_func == 'class_balanced_cross_entropy':
        return class_balanced_cross_entropy_loss(outputs, gts, **loss_kwargs)
    elif loss_func == 'dice':
        return dice_loss(outputs, gts, **loss_kwargs)
    elif loss_func == 'cross_entropy_and_dice':
        reduction = 'mean'
        if 'batch_average' in loss_kwargs and not loss_kwargs['batch_average']:
            reduction = 'none'
        criterion = nn.BCEWithLogitsLoss(reduction=reduction).cuda()
        loss = criterion(outputs, gts)
        if reduction == 'none':
            loss = loss.view(loss.shape[0], -1).mean(dim=1)

        return loss - (1 - dice_loss(outputs, gts, **loss_kwargs)).log()
    else:
        raise NotImplementedError


def epoch_iter(num_epochs: int):
    # one epoch corresponds to one random transformed first frame of a sequence
    if num_epochs is None:
        return count(start=1)
    else:
        return range(1, num_epochs + 1)


def run_loader(model, loader, loss_func, img_save_dir=None, return_probs=False):
    device = next(model.parameters()).device

    metrics = {n: [] for n in ['loss_batches', 'acc_batches']}

    # if hasattr(loader.sampler, 'indices') and loader.sampler.indices is not None:
    #     assert 1 in loader.sampler.indices
    loader_frame_id = loader.dataset.frame_id
    # loader.dataset.frame_id = None
    loader.dataset.set_gt_frame_id()
    train_frame = loader.dataset[loader.dataset.frame_id]
    train_frame_gt = train_frame['gt']
    loader.dataset.frame_id = loader_frame_id
    targets = train_frame_gt.unsqueeze(dim=0)

    probs_all = []
    boxes_all =[]
    with torch.no_grad():
        for sample_batched in loader:
            imgs, gts, file_names = sample_batched['image'], sample_batched['gt'], sample_batched['file_name']
            inputs, gts = imgs.to(device), gts.to(device)

            model.eval()

            # targets = gts

            if isinstance(model, MaskRCNN):
                outputs = model(inputs, targets)

                probs = outputs[0]
                targets = probs.ge(0.5).float()
                # model.rpn.augment_target_proposals_mode = 'REPLACE'
                if targets.sum().item() == 0:
                    # model.rpn.augment_target_proposals_mode = 'EXTEND'
                    targets = train_frame_gt.unsqueeze(dim=0)

                metrics['loss_batches'].append(torch.tensor([0.0]))

                boxes_all.append(outputs[1])
            else:
                outputs = model(inputs)
                probs = torch.sigmoid(outputs[-1])

                loss = compute_loss(loss_func, outputs[-1], gts, {'batch_average': False})
                metrics['loss_batches'].append(loss)

            preds = probs.ge(0.5).float()
            probs_all.append(probs)
            # print(preds.eq(gts.bool()).view(preds.size(0), -1).sum(dim=1).float().div(preds[0].numel()).shape)
            metrics['acc_batches'].append(preds.bool().eq(gts.bool()).view(preds.size(0), -1).sum(dim=1).float().div(preds[0].numel()))

            if img_save_dir is not None:
                # preds = 1 * preds
                preds = np.transpose(preds.cpu().numpy(), (0, 2, 3, 1)).astype(np.uint8)

                if loader.dataset.flip_label:
                    preds = np.logical_not(preds).astype(np.uint8)

                for file_name, pred in zip(file_names, preds):
                    pred_path = os.path.join(img_save_dir, os.path.basename(file_name) + '.png')
                    imageio.imsave(pred_path, pred)

    metrics = {n: torch.cat(m).cpu() for n, m in metrics.items()}
    if return_probs:
        return metrics['loss_batches'], metrics['acc_batches'], torch.cat(probs_all), torch.cat(boxes_all)
    return metrics['loss_batches'], metrics['acc_batches']


def eval_loader(model, loader, loss_func, img_save_dir=None, return_preds=False):
    seq_name = loader.dataset.seq_key

    if img_save_dir is None:
        img_save_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(img_save_dir, seq_name))

    loss_batches, acc_batches, preds, _ = run_loader(model, loader, loss_func, os.path.join(img_save_dir, seq_name), True)

    evaluation = eval_davis_seq(img_save_dir, seq_name)

    if '/tmp/' in img_save_dir:
        shutil.rmtree(img_save_dir)
    if return_preds:
        return loss_batches, acc_batches, evaluation['J']['mean'], evaluation['F']['mean'], preds
    return loss_batches, acc_batches, evaluation['J']['mean'], evaluation['F']['mean']


def train_val(model, train_loader, val_loader, optim, num_epochs,
              seed, _log, early_stopping_func=None,
              validate_inter=None, loss_func='cross_entropy',
              lr_scheduler=None):
    device = next(model.parameters()).device

    metrics_names = ['train_loss', 'val_loss', 'val_J', 'val_F', 'val_acc']
    metrics = {n: [] for n in metrics_names}

    ave_grad = 0

    if early_stopping_func is None:
        early_stopping_func = lambda loss_hist: False

    for epoch in epoch_iter(num_epochs):
        set_random_seeds(seed + epoch)
        for _, sample_batched in enumerate(train_loader):
            inputs, gts = sample_batched['image'], sample_batched['gt']
            inputs, gts = inputs.to(device), gts.to(device)

            model.train_without_dropout()

            if isinstance(model, MaskRCNN):
                train_loss = model(inputs, gts)[0]
            else:
                outputs = model(inputs)
                train_loss = compute_loss(loss_func, outputs[-1], gts)

            metrics['train_loss'].append(train_loss.item())

            ave_grad += 1

            if isinstance(optim, MetaOptimizer):
                optim.set_train_loss(train_loss)
                optim.step(train_loss)
            else:
                train_loss.backward()
                optim.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            model.zero_grad()
            ave_grad = 0

            if validate_inter is not None and epoch % validate_inter == 0:
                val_loss_batches, val_acc_batches, val_J, val_F = eval_loader(model, val_loader, loss_func)
                metrics['val_loss'].append(val_loss_batches.mean())
                metrics['val_acc'].append(val_acc_batches.mean())
                metrics['val_J'].append(val_J)
                metrics['val_F'].append(val_F)

            if early_stopping_func(metrics['train_loss']):
                break

        if early_stopping_func(metrics['train_loss']):
            break

    metrics = {n: torch.tensor(m) for n, m in metrics.items()}
    return metrics['train_loss'], metrics['val_loss'], metrics['val_acc'], metrics['val_J'], metrics['val_F']


def data_loaders(dataset, random_train_transform, batch_sizes, shuffles,
                 frame_ids, num_workers, crop_sizes, multi_object, pin_memory,
                 normalize, full_resolution=False):
    # train
    train_transforms = []
    if random_train_transform:
        train_transforms.extend([custom_transforms.RandomHorizontalFlip(),
                                 custom_transforms.RandomScaleNRotate(rots=(-30, 30),
                                                                      scales=(.75, 1.25))])
    train_transforms.append(custom_transforms.ToTensor())
    composed_transforms = transforms.Compose(train_transforms)

    if dataset['name'] == 'DAVIS-2016':
        vos_dataset = DAVIS
        root_dir = 'data/DAVIS-2016'
    elif dataset['name'] == 'DAVIS-2017':
        vos_dataset = DAVIS
        root_dir = 'data/DAVIS-2017'
    elif dataset['name'] == 'YouTube-VOS':
        vos_dataset = YouTube
        root_dir = 'data/YouTube-VOS'
    else:
        raise NotImplementedError

    db_train = vos_dataset(
        root_dir=root_dir,
        seqs_key=dataset['split'],
        frame_id=frame_ids['train'],
        transform=composed_transforms,
        crop_size=crop_sizes['train'],
        multi_object=multi_object,
        normalize=normalize,
        full_resolution=full_resolution)

    # sample epochs into a batch
    batch_sampler = EpochSampler(
        db_train, shuffles['train'], batch_sizes['train'])
    train_loader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory)

    # test
    db_test = vos_dataset(
        root_dir=root_dir,
        seqs_key=dataset['split'],
        frame_id=frame_ids['test'],
        transform=custom_transforms.ToTensor(),
        crop_size=crop_sizes['test'],
        multi_object=multi_object,
        normalize=normalize,
        full_resolution=full_resolution)
    test_loader = DataLoader(
        db_test,
        shuffle=shuffles['test'],
        batch_size=batch_sizes['test'],
        num_workers=num_workers,
        sampler=SequentialSubsetSampler(db_test),
        pin_memory=pin_memory)

    if 'meta' not in batch_sizes:
        return train_loader, test_loader

    # meta
    db_meta = vos_dataset(
        root_dir=root_dir,
        seqs_key=dataset['split'],
        frame_id=frame_ids['meta'],
        transform=custom_transforms.ToTensor(),
        crop_size=crop_sizes['meta'],
        multi_object=multi_object,
        normalize=normalize,
        full_resolution=full_resolution)

    meta_loader = DataLoader(
        db_meta,
        shuffle=shuffles['meta'],
        batch_size=batch_sizes['meta'],
        num_workers=num_workers,
        sampler=SequentialSubsetSampler(db_meta),
        pin_memory=pin_memory)

    return train_loader, test_loader, meta_loader


def init_parent_model(architecture, encoder, train_encoder, decoder_norm_layer,
                      replace_batch_with_group_norms, batch_norm,
                      roi_pool_output_sizes, eval_augment_rpn_proposals_mode,
                      **datasets):
    if architecture == 'FPN':
        model = FPN(encoder, classes=1, activation=None, decoder_dropout=0.0,
                    batch_norm=batch_norm, train_encoder=train_encoder,
                    decoder_norm_layer=decoder_norm_layer)
    elif architecture == 'DeepLabV3':
        model = DeepLabV3(encoder, num_classes=1, batch_norm=batch_norm, train_encoder=train_encoder)
    elif architecture == 'DeepLabV3Plus':
        model = DeepLabV3Plus(
            encoder, num_classes=1, batch_norm=batch_norm, train_encoder=train_encoder,
            replace_batch_with_group_norms=replace_batch_with_group_norms)
    elif architecture == 'MaskRCNN':
        model = MaskRCNN(
            encoder, num_classes=2, batch_norm=batch_norm, train_encoder=train_encoder,
            roi_pool_output_sizes=roi_pool_output_sizes,
            eval_augment_rpn_proposals_mode=eval_augment_rpn_proposals_mode,
            replace_batch_with_group_norms=replace_batch_with_group_norms)
    else:
        raise NotImplementedError

    parent_states = {}
    for k, v in datasets.items():
        parent_states[k] = {}
        states = [torch.load(p, map_location=lambda storage, loc: storage)
                             for p in v['paths']]

        # states = [{k: state[k] if k in state else v
        #            for k, v in model.state_dict().items()}
        #           for state in states]

        parent_states[k]['states'] = states
        parent_states[k]['splits'] = [np.loadtxt(p, dtype=str).tolist()
                                      for p in v['val_split_files']]

    # model.load_state_dict(parent_states['train']['states'][0])
    # model.merge_batch_norms_with_convs()
    # import copy
    # parent_states['train']['states'][0] = copy.deepcopy(model.state_dict())
    # parent_states['val']['states'][0] = copy.deepcopy(model.state_dict())

    # state_dict = model.state_dict()
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         parent_states['train']['states'][0][n] = state_dict[n]
    # parent_states['train']['states'][0] = model.state_dict()

    return model, parent_states


def early_stopping(loss_hist, patience, min_loss_improv):
    if patience is None or len(loss_hist) <= patience:
        return False

    best_loss = torch.tensor(loss_hist).min()
    prev_best_loss = torch.tensor(loss_hist[:-patience]).min()

    if torch.gt(best_loss.sub(prev_best_loss).abs(), min_loss_improv):
        return False
    return True


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def eval_davis(results_dir, seq_name):
    metrics = ['J', 'F']
    statistics = ['mean', 'recall', 'decay']

    if seq_name == 'train_seqs':
        phase = davis.phase['TRAIN']
    elif seq_name == 'test_seqs':
        phase = davis.phase['VAL']

    db = DAVISLoader('2016', phase, True)

    # Load segmentations
    segmentations = [Segmentation(os.path.join(results_dir, s), True)
                     for s in db.iternames()]

    # Evaluate results
    evaluation = db_eval(db, segmentations, metrics)

    # Print results
    table = PrettyTable(['Method']+[p[0]+'_'+p[1] for p in
                                    product(metrics, statistics)])

    table.add_row([os.path.basename(results_dir)] + ["%.3f" % np.round(
        evaluation['dataset'][metric][statistic], 3) for metric, statistic
        in product(metrics, statistics)])

    return evaluation, table


def eval_davis_seq(results_dir, seq_name):
    # TODO: refactor
    from davis import cfg as eval_cfg
    segmentations = Segmentation(os.path.join(
        results_dir, seq_name), not eval_cfg.MULTIOBJECT)
    annotations = Annotation(seq_name, not eval_cfg.MULTIOBJECT)

    evaluation = {}
    for m in ['J', 'F']:
        evaluation[m] = db_eval_sequence(segmentations, annotations, measure=m)
    return evaluation


class SequentialSubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = indices

    def __iter__(self):
        if self.indices is None:
            return iter(range(len(self.dataset)))
        return (i for i in self.indices)

    def __len__(self):
        if self.indices is None:
            return len(self.dataset)
        return len(self.indices)
