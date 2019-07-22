import os
import shutil
import tempfile
from itertools import count, product, zip_longest

import davis
import imageio
import numpy as np
import torch
from dataloaders import custom_transforms
from dataloaders import davis_2016 as db
from davis import (Annotation, DAVISLoader, Segmentation, db_eval,
                   db_eval_sequence)
from layers.osvos_layers import class_balanced_cross_entropy_loss, dice_loss
from networks.drn_seg import DRNSeg
from networks.unet import Unet
from networks.fpn import FPN
from networks.vgg_osvos import OSVOSVgg
from prettytable import PrettyTable
from pytorch_tools.data import EpochSampler
from pytorch_tools.ingredients import set_random_seeds
from torch.utils.data import DataLoader
from torchvision import transforms


def run_loader(model, loader, loss_func, img_save_dir=None):
    device = next(model.parameters()).device
    
    metrics = {n: [] for n in ['loss_batches', 'acc_batches']}

    with torch.no_grad():
        for sample_batched in loader:
            imgs, gts, fnames = sample_batched['image'], sample_batched['gt'], sample_batched['fname']
            inputs, gts = imgs.to(device), gts.to(device)
            outputs = model.forward(inputs)

            if loss_func == 'cross_entropy':
                loss = class_balanced_cross_entropy_loss(outputs[-1], gts)
            elif loss_func == 'dice':
                loss = dice_loss(outputs[-1], gts)
            else:
                raise NotImplementedError

            metrics['loss_batches'].append(loss.item())

            preds = torch.sigmoid(outputs[-1])
            preds = preds >= 0.5
            metrics['acc_batches'].append(preds.eq(gts.byte()).sum().float().div(preds.numel()).item())

            if img_save_dir is not None:
                preds = 255 * preds
                preds = np.transpose(preds.cpu().numpy(), (0, 2, 3, 1)).astype(np.uint8)

                for fname, pred in zip(fnames, preds):
                    pred_path = os.path.join(img_save_dir, os.path.basename(fname) + '.png')
                    imageio.imsave(pred_path, pred)

    # metrics = {n: torch.tensor(m).mean().item() for n, m in metrics.items()}
    metrics = {n: torch.tensor(m) for n, m in metrics.items()}
    return metrics['loss_batches'], metrics['acc_batches']


def eval_loader(model, loader, loss_func, img_save_dir=None):
    seq_name = loader.dataset.seqs

    if img_save_dir is None:
        img_save_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(img_save_dir, seq_name))

    loss_batches, acc_batches = run_loader(model, loader, loss_func, os.path.join(img_save_dir, seq_name))
    
    evaluation = eval_davis_seq(img_save_dir, seq_name)

    if '/tmp/' in img_save_dir:
        shutil.rmtree(img_save_dir)
    
    return loss_batches, acc_batches, evaluation['J']['mean'][0], evaluation['F']['mean'][0]


def train_val(model, train_loader, val_loader, optim, num_epochs,
              seed, _log, early_stopping_func=None,
              validate_inter=None, loss_func='cross_entropy'):
    device = next(model.parameters()).device

    metrics_names = ['train_loss', 'val_loss', 'val_J', 'val_F', 'val_acc']
    metrics = {n: [] for n in metrics_names}

    ave_grad = 0

    if early_stopping_func is None:
        early_stopping_func = lambda loss_hist: False

    # if _log is not None:
    #     seq_name = train_loader.dataset.seqs
    #     _log.info(f"Train OSVOS online - SEQUENCE: {seq_name}")

    if num_epochs is None:
        epoch_iter = count()
    else:
        epoch_iter = range(1, num_epochs + 1)

    for epoch in epoch_iter:
        set_random_seeds(seed + epoch)
        for _, sample_batched in enumerate(train_loader):
            inputs, gts = sample_batched['image'], sample_batched['gt']
            inputs, gts = inputs.to(device), gts.to(device)

            outputs = model(inputs)

            if loss_func == 'cross_entropy':
                loss = class_balanced_cross_entropy_loss(outputs[-1], gts)
            elif loss_func == 'dice':
                loss = dice_loss(outputs[-1], gts)
            else:
                raise NotImplementedError
            metrics['train_loss'].append(loss.item())

            loss.backward()
            ave_grad += 1

            optim.step()
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


def datasets_and_loaders(seq_name, random_train_transform, batch_sizes,
                         shuffles, frame_ids):
    train_transforms = []
    if random_train_transform:
        train_transforms.extend([custom_transforms.RandomHorizontalFlip(),
                                 custom_transforms.ScaleNRotate(rots=(-30, 30),
                                                                scales=(.75, 1.25))])
    train_transforms.append(custom_transforms.ToTensor())
    composed_transforms = transforms.Compose(train_transforms)

    db_train = db.DAVIS2016(seqs=seq_name,
                            frame_id=frame_ids['train'],
                            transform=composed_transforms)
    batch_sampler = EpochSampler(
        db_train, shuffles['train'], batch_sizes['train'])
    train_loader = DataLoader(
        db_train, batch_sampler=batch_sampler, num_workers=0)

    db_test = db.DAVIS2016(seqs=seq_name,
                           frame_id=frame_ids['test'],
                           transform=custom_transforms.ToTensor())
    test_loader = DataLoader(
        db_test,
        shuffle=shuffles['test'],
        batch_size=batch_sizes['test'],
        num_workers=0)

    return db_train, train_loader, db_test, test_loader


def init_parent_model(cfg):
    base_path = cfg['base_path']
    if 'VGG' in base_path:
        model = OSVOSVgg(pretrained=0)
    elif 'DRN_D_22' in base_path:
        model = DRNSeg('DRN_D_22', 1, pretrained=True)
    elif 'UNET_ResNet18' in base_path:
        model = Unet('resnet18', classes=1, activation='softmax')
    elif 'UNET_ResNet34' in base_path:
        model = Unet('resnet34', classes=1, activation='softmax')
    elif 'FPN_ResNet34' in base_path:
        model = FPN('resnet34', classes=1, activation='softmax', dropout=0.0)
    else:
        raise NotImplementedError

    parent_state_dicts = []
    for p in cfg['split_model_path']:
        split_model_path = os.path.join(cfg['base_path'], p)
        parent_state_dicts.append(torch.load(
            split_model_path, map_location=lambda storage, loc: storage))
    # model.load_state_dict(parent_state_dict)
    return model, parent_state_dicts


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
    segmentations = Segmentation(os.path.join(results_dir, seq_name), True)
    annotations = Annotation(seq_name, True)

    evaluation = {}
    for m in ['J', 'F']:
        evaluation[m] = db_eval_sequence(segmentations, annotations, measure=m)
    return evaluation
