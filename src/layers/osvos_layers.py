from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F


def logit(x):
    return np.log(x/(1-x+1e-08)+1e-08)


def sigmoid_np(x):
    return 1/(1+np.exp(-x))


def class_balanced_cross_entropy_loss(output, label, size_average=False, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = torch.ge(label, 0.5).float()

    # TODO: refactor
    if not batch_average:
        batch_size = output.size(0)
        num_labels_pos = torch.sum(labels.view(batch_size, -1), dim=1, keepdim=True)
        num_labels_neg = torch.sum(1.0 - labels.view(batch_size, -1), dim=1, keepdim=True)
        num_total = num_labels_pos + num_labels_neg

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos = torch.sum(-torch.mul(labels, loss_val).view(batch_size, -1), dim=1, keepdim=True)
        loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val).view(batch_size, -1), dim=1, keepdim=True)

        final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

        if size_average:
            final_loss /= np.prod(label.size())
    else:
        num_labels_pos = torch.sum(labels)
        num_labels_neg = torch.sum(1.0 - labels)
        num_total = num_labels_pos + num_labels_neg

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos = torch.sum(-torch.mul(labels, loss_val))
        loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

        final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

        final_loss /= label.size()[0]

    if size_average:
        final_loss /= np.prod(label.size())

    return final_loss


def class_balanced_cross_entropy_loss_theoretical(output, label, size_average=True, batch_average=True):
    """Theoretical version of the class balanced cross entropy loss to train the network (Produces unstable results)
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """
    output = torch.sigmoid(output)

    labels_pos = torch.ge(label, 0.5).float()
    labels_neg = torch.lt(label, 0.5).float()

    num_labels_pos = torch.sum(labels_pos)
    num_labels_neg = torch.sum(labels_neg)
    num_total = num_labels_pos + num_labels_neg

    # loss_pos = torch.mul(labels_pos, torch.log(output + 1e-8))
    # loss_neg = torch.mul(labels_neg, torch.log(1 - output + 1e-8))
    loss_pos = torch.sum(torch.mul(labels_pos, torch.log(output + 1e-8)))
    loss_neg = torch.sum(torch.mul(labels_neg, torch.log(1 - output + 1e-8)))

    final_loss = -num_labels_neg / num_total * \
        loss_pos - num_labels_pos / num_total * loss_neg

    # final_loss = - loss_pos - loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss


def dice_loss(output, label, batch_average=True):
    pred = torch.sigmoid(output)
    smooth = 1.

    if len(torch.unique(label)) > 2:
        raise NotImplementedError
    # unlabeled_mask = label.eq(255)
    # label[unlabeled_mask] = 0
    # output[unlabeled_mask] = 0

    # TODO: refactor
    if batch_average:
        pred_flat = pred.view(-1)
        label_flat = label.view(-1)
        intersection = pred_flat * label_flat

        return 1 - ((2. * intersection.sum() + smooth) /
                    (pred_flat.sum() + label_flat.sum() + smooth))
    else:
        batch_dim = pred.size(0)

        pred_flat = pred.view(batch_dim, -1)
        label_flat = label.view(batch_dim, -1)
        intersection = pred_flat * label_flat


        return 1 - ((2. * intersection.sum(dim=1) + smooth) /
                    (pred_flat.sum(dim=1) + label_flat.sum(dim=1) + smooth))


def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)

    # fixed indexing for PyTorch 0.4
    return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# this is for deconvolution without groups
def interp_surgery(lay):
        m, k, h, w = lay.weight.data.size()
        if m != k:
            print('input + output channels need to be the same')
            raise ValueError
        if h != w:
            print('filters need to be square')
            raise ValueError
        filt = upsample_filt(h)

        for i in range(m):
            lay.weight[i, i, :, :].data.copy_(torch.from_numpy(filt))

        return lay.weight.data


if __name__ == '__main__':
    import os
    from mypath import Path

    # Output
    output = Image.open(os.path.join(Path.db_root_dir(), 'Annotations/480p/blackswan/00000.png'))
    output = np.asarray(output, dtype=np.float32)/255.0
    output = logit(output)
    output = Variable(torch.from_numpy(output)).cuda()

    # GroundTruth
    label = Image.open(os.path.join(Path.db_root_dir(), 'Annotations/480p/blackswan/00001.png'))
    label = Variable(torch.from_numpy(np.asarray(label, dtype=np.float32))/255.0).cuda()

    loss = class_balanced_cross_entropy_loss(output, label)
    print(loss)
