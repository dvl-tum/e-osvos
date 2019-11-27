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
from data import DAVIS, custom_transforms
from davis import (Annotation, DAVISLoader, Segmentation, db_eval,
                   db_eval_sequence)
from layers.osvos_layers import class_balanced_cross_entropy_loss, dice_loss
from meta_optim.meta_optim import MetaOptimizer
from networks.drn_seg import DRNSeg
from networks.fpn import FPN
from networks.unet import Unet
from networks.vgg_osvos import OSVOSVgg
# from networks.deeplab import DeepLab
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
        return class_balanced_cross_entropy_loss(outputs, gts, **loss_kwargs)
    elif loss_func == 'dice':
        return dice_loss(outputs, gts, **loss_kwargs)
    else:
        raise NotImplementedError


def epoch_iter(num_epochs: int):
    # one epoch corresponds to one random transformed first frame of a sequence
    if num_epochs is None:
        return count(start=1)
    else:
        return range(1, num_epochs + 1)


def run_loader(model, loader, loss_func, img_save_dir=None, return_preds=False):
    device = next(model.parameters()).device

    metrics = {n: [] for n in ['loss_batches', 'acc_batches']}

    preds_all = []
    with torch.no_grad():
        for sample_batched in loader:
            imgs, gts, file_names = sample_batched['image'], sample_batched['gt'], sample_batched['file_name']
            inputs, gts = imgs.to(device), gts.to(device)

            model.eval()
            outputs = model.forward(inputs)

            loss = compute_loss(loss_func, outputs[-1], gts, {'batch_average': False})
            metrics['loss_batches'].append(loss)

            preds = torch.sigmoid(outputs[-1])
            preds = preds.ge(0.5).float()
            preds_all.append(preds)
            # print(preds.eq(gts.bool()).view(preds.size(0), -1).sum(dim=1).float().div(preds[0].numel()).shape)
            metrics['acc_batches'].append(preds.bool().eq(gts.bool()).view(preds.size(0), -1).sum(dim=1).float().div(preds[0].numel()))

            if img_save_dir is not None:
                # preds = 1 * preds
                preds = np.transpose(preds.cpu().numpy(), (0, 2, 3, 1)).astype(np.uint8)

                for file_name, pred in zip(file_names, preds):
                    pred_path = os.path.join(img_save_dir, os.path.basename(file_name) + '.png')
                    imageio.imsave(pred_path, pred)

    metrics = {n: torch.cat(m).cpu() for n, m in metrics.items()}
    if return_preds:
        return metrics['loss_batches'], metrics['acc_batches'], torch.cat(preds_all)
    return metrics['loss_batches'], metrics['acc_batches']


def eval_loader(model, loader, loss_func, img_save_dir=None, return_preds=False):
    seq_name = loader.dataset.seq_key

    if img_save_dir is None:
        img_save_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(img_save_dir, seq_name))

    loss_batches, acc_batches, preds = run_loader(model, loader, loss_func, os.path.join(img_save_dir, seq_name), True)

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

            model.train()
            outputs = model(inputs)

            train_loss = compute_loss(loss_func, outputs[-1], gts)
            metrics['train_loss'].append(train_loss.item())

            train_loss.backward()
            ave_grad += 1

            if isinstance(optim, MetaOptimizer):
                optim.set_train_loss(train_loss)

            # if optim is a model
            with torch.no_grad():
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


def data_loaders(dataset, root_dir, random_train_transform, batch_sizes,
                 shuffles, frame_ids, num_workers, crop_sizes, multi_object,
                 pin_memory):
    # train
    train_transforms = []
    if random_train_transform:
        train_transforms.extend([custom_transforms.RandomHorizontalFlip(),
                                 custom_transforms.RandomScaleNRotate(rots=(-30, 30),
                                                                      scales=(.75, 1.25))])
    train_transforms.append(custom_transforms.ToTensor())
    composed_transforms = transforms.Compose(train_transforms)

    db_train = DAVIS(
        root_dir=root_dir,
        seqs_key=dataset,
        frame_id=frame_ids['train'],
        transform=composed_transforms,
        crop_size=crop_sizes['train'],
        multi_object=multi_object,)

    # sample epochs into a batch
    batch_sampler = EpochSampler(
        db_train, shuffles['train'], batch_sizes['train'])
    train_loader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory)

    # test
    db_test = DAVIS(
        root_dir=root_dir,
        seqs_key=dataset,
        frame_id=frame_ids['test'],
        transform=custom_transforms.ToTensor(),
        crop_size=crop_sizes['test'],
        multi_object=multi_object,)
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
    db_meta = DAVIS(
        root_dir=root_dir,
        seqs_key=dataset,
        frame_id=frame_ids['meta'],
        transform=custom_transforms.ToTensor(),
        crop_size=crop_sizes['meta'],
        multi_object=multi_object,)

    meta_loader = DataLoader(
        db_meta,
        shuffle=shuffles['meta'],
        batch_size=batch_sizes['meta'],
        num_workers=num_workers,
        pin_memory=pin_memory)

    return train_loader, test_loader, meta_loader


def init_parent_model(architecture, encoder, train_encoder, decoder_norm_layer, batch_norm, **datasets):
    # if 'VGG' in base_path:
    #     model = OSVOSVgg(pretrained=0)
    # elif 'DRN_D_22' in base_path:
    #     model = DRNSeg('DRN_D_22', 1, pretrained=True)
    # elif 'UNET_ResNet18' in base_path:
    #     model = Unet('resnet18', classes=1, activation='softmax')
    # elif 'FPN_ResNet34_group_norm' in base_path:
    #     model = FPN('resnet34-group-norm', classes=1, activation='softmax', dropout=0.0)
    # elif 'UNET_ResNet34' in base_path:
    #     model = Unet('resnet34', classes=1, activation='softmax')
    # elif 'DeepLab_ResNet101' in parent_model_path:
    #     model = DeepLab(backbone='resnet', output_stride=16, num_classes=1, freeze_bn=True)

    if architecture == 'FPN':
        model = FPN(encoder, classes=1, activation='softmax',
                    dropout=0.0, batch_norm=batch_norm,
                    train_encoder=train_encoder, decoder_norm_layer=decoder_norm_layer)
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

    # if 'DeepLab_ResNet101' in parent_model_path:
    #     parent_state_dict = parent_state_dict['state_dict']
    #     parent_state_dict['decoder.last_conv.8.weight'] = model.state_dict()['decoder.last_conv.8.weight'].clone()
    #     parent_state_dict['decoder.last_conv.8.bias'] = model.state_dict()['decoder.last_conv.8.bias'].clone()
    # model.load_state_dict(parent_state_dict)
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
