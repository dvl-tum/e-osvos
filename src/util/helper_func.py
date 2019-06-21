import os
from itertools import count, zip_longest

import imageio
import numpy as np
import torch
from dataloaders import custom_transforms
from dataloaders import davis_2016 as db
from layers.osvos_layers import class_balanced_cross_entropy_loss
from networks.drn_seg import DRNSeg
from networks.vgg_osvos import OSVOSVgg
from networks.unet_resnet import Unet
from pytorch_tools.data import EpochSampler
from pytorch_tools.ingredients import set_random_seeds
from torch.utils.data import DataLoader
from torchvision import transforms


def run_loader(model, loader, img_save_dir=None):
    device = next(model.parameters()).device
    run_loss = []
    with torch.no_grad():
        for sample_batched in loader:
            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']
            inputs, gts = img.to(device), gt.to(device)
            outputs = model.forward(inputs)

            loss = class_balanced_cross_entropy_loss(
                outputs[-1], gts, size_average=False)
            run_loss.append(loss.item())

            if img_save_dir is not None:
                for jj in range(int(inputs.size()[0])):
                    pred = np.transpose(
                        outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                    pred = 1 / (1 + np.exp(-pred))
                    pred = 255 * np.squeeze(pred)
                    pred = pred.astype(np.uint8)

                    imageio.imsave(os.path.join(
                        img_save_dir, os.path.basename(fname[jj]) + '.png'), pred)

    return torch.tensor(run_loss).mean()


def train_val(model, train_loader, val_loader, optim, num_epochs,
              num_ave_grad, seed, _log, early_stopping_func=None):
    device = next(model.parameters()).device

    train_loss_hist = []
    val_losses = []
    ave_grad = 0

    if early_stopping_func is None:
        early_stopping_func = lambda loss_hist: False

    if _log is not None:
        seq_name = train_loader.dataset.seqs
        _log.info(f"Train OSVOS online - SEQUENCE: {seq_name}")

    if num_epochs is None:
        epoch_iter = count()
    else:
        epoch_iter = range(num_epochs * num_ave_grad)
    for epoch in epoch_iter:
        set_random_seeds(seed + epoch)
        for _, sample_batched in enumerate(train_loader):
            inputs, gts = sample_batched['image'], sample_batched['gt']
            inputs, gts = inputs.to(device), gts.to(device)

            outputs = model(inputs)

            # Compute the fuse loss
            loss = class_balanced_cross_entropy_loss(
                outputs[-1], gts, size_average=False)
            train_loss_hist.append(loss.item())

            loss /= num_ave_grad
            loss.backward()
            ave_grad += 1

            # Update the weights once in num_ave_grad forward passes
            if ave_grad % num_ave_grad == 0:
                optim.step()
                model.zero_grad()
                ave_grad = 0

                if val_loader is not None:
                    val_loss = run_loader(model, val_loader)
                    val_losses.append(val_loss.item())

            if early_stopping_func(train_loss_hist):
                break

        if early_stopping_func(train_loss_hist):
            break

    if _log is not None:
        _log.info(
            f'RUN TRAIN loss: {torch.tensor(train_loss_hist).mean().item():.2f}')
        if val_loader is not None:
            _log.info(f'LAST VAL loss: {val_losses[-1]:.2f}')

            best_val_epoch = torch.tensor(val_losses).argmin()
            best_val_loss = torch.tensor(val_losses)[best_val_epoch]
            _log.info(f'BEST VAL loss/epoch: {best_val_loss:.2f}/{best_val_epoch + 1}')
    return torch.tensor(train_loss_hist).mean(), val_losses


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

    if torch.ge(best_loss.sub(prev_best_loss).abs(), min_loss_improv):
        return False
    return True


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
