import torch


def dice_loss(output, label, batch_average=True):
    pred = torch.sigmoid(output)
    smooth = 1.

    for l in torch.unique(label):
        if l not in [0.0, 1.0]:
            raise NotImplementedError
    if len(torch.unique(label)) > 2:
        raise NotImplementedError

    # # label must be foreground/background plus additional non-labeled label
    # # label must be torch.float and normalized
    # if len(torch.unique(label)) > 2 or label.gt(1.0).any():
    #     raise NotImplementedError
    # elif len(torch.unique(label)) == 3:
    #     unlabeled_mask = label.eq(1.0)
    #     label[unlabeled_mask] = 0
    #     output[unlabeled_mask] = 0

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
