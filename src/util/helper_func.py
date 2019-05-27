import os

import imageio
import numpy as np
import torch
from layers.osvos_layers import class_balanced_cross_entropy_loss
from pytorch_tools.ingredients import set_random_seeds


def run_loader(model, loader, save_dir=None):
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

            if save_dir is not None:
                for jj in range(int(inputs.size()[0])):
                    pred = np.transpose(
                        outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                    pred = 1 / (1 + np.exp(-pred))
                    pred = 255 * np.squeeze(pred)
                    pred = pred.astype(np.uint8)

                    imageio.imsave(os.path.join(
                        save_dir, os.path.basename(fname[jj]) + '.png'), pred)

    return torch.tensor(run_loss).mean()


def train_test(model, train_loader, test_loader, optimizer, num_epochs,
               num_ave_grad, seed, _log):
    device = next(model.parameters()).device

    test_losses = []
    run_train_loss = []
    ave_grad = 0

    if _log is not None:
        seq_name = train_loader.dataset.seqs
        _log.info(f"Train regular OSVOS - SEQUENCE: {seq_name}")
    for epoch in range(0, num_epochs * num_ave_grad):
        set_random_seeds(seed + epoch)
        for _, sample_batched in enumerate(train_loader):
            inputs, gts = sample_batched['image'], sample_batched['gt']
            inputs, gts = inputs.to(device), gts.to(device)

            outputs = model(inputs)

            # Compute the fuse loss
            loss = class_balanced_cross_entropy_loss(
                outputs[-1], gts, size_average=False)
            run_train_loss.append(loss.item())

            loss /= num_ave_grad
            loss.backward()
            ave_grad += 1

            # Update the weights once in num_ave_grad forward passes
            if ave_grad % num_ave_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
                ave_grad = 0

                if test_loader is not None:
                    test_loss = run_loader(model, test_loader)
                    test_losses.append(test_loss.item())

    if _log is not None:
        _log.info(
            f'RUN TRAIN loss: {torch.tensor(run_train_loss).mean().item():.2f}')
        if test_loader is not None:
            best_test_epoch = torch.tensor(test_losses).argmin()
            best_test_loss = torch.tensor(test_losses)[best_test_epoch]
            _log.info(
                f'BEST TEST loss/epoch: {best_test_loss:.2f}/{best_test_epoch + 1}')
    return torch.tensor(run_train_loss).mean(), test_losses
