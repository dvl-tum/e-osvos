import os

import imageio
import numpy as np
import torch
from layers.osvos_layers import class_balanced_cross_entropy_loss


def run_loader(model, loader, save_dir=None, seq_name=None):
    if save_dir is not None and seq_name is not None:
        save_dir_res = os.path.join(save_dir, 'results', seq_name)
        if not os.path.exists(save_dir_res):
            os.makedirs(save_dir_res)

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

            if save_dir is not None and seq_name is not None:
                for jj in range(int(inputs.size()[0])):
                    pred = np.transpose(
                        outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                    pred = 1 / (1 + np.exp(-pred))
                    pred = 255 * np.squeeze(pred)
                    pred = pred.astype(np.uint8)

                    imageio.imsave(os.path.join(
                        save_dir_res, os.path.basename(fname[jj]) + '.png'), pred)

    return torch.tensor(run_loss).mean()
