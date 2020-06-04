import types
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import MaskRCNN as _MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.roi_heads import project_masks_on_boxes, maskrcnn_inference, fastrcnn_loss, keypointrcnn_loss, keypointrcnn_inference
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops import boxes as box_ops
from .lovasz_losses import lovasz_hinge

# from .efficient_det.models.bifpn import BIFPN as _BIFPN
# from .efficient_det.models.efficientnet import EfficientNet

from .efficient_det_v2.efficientdet.model import BiFPN as _BiFPN
from .efficient_det_v2.efficientdet.model import EfficientNet as _EfficientNet
from .efficient_det_v2.efficientnet.utils_extra import Conv2dStaticSamePadding


class EfficientNet(_EfficientNet):

    def forward(self, x):
        feature_maps = super(EfficientNet, self).forward(x)
        return feature_maps[-3:]
        # print([f.shape for f in feature_maps])
        # return feature_maps


class EfficientNetAndBiFPN(nn.Sequential):

    def __init__(self, efficient_net, bifpn):
        super(EfficientNetAndBiFPN, self).__init__(
            efficient_net,
            bifpn)

    def forward(self, x):
        outs = super(EfficientNetAndBiFPN, self).forward(x)
        names = [0, 1, 2, 3, 'pool']
        return OrderedDict([(n, o) for n, o in zip(names, outs)])


class BiFPN(_BiFPN):

#     def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
#         super(BiFPN, self).__init__(num_channels, conv_channels, first_time, epsilon, onnx_export, attention)

#         if self.first_time:
#             del self.p5_to_p6

#             self.p6_down_channel = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

            # p3, p4, p5, p6 = inputs

            # p6_in = self.p5_to_p6(p5)


            # p3_in = self.p3_down_channel(p3)
            # p4_in = self.p4_down_channel(p4)
            # p5_in = self.p5_down_channel(p5)
            # p6_in = self.p6_down_channel(p6)

            # p7_in = self.p6_to_p7(p6_in)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        # p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        scale_factor = [p6_in.shape[2] / p7_in.shape[2],
                        p6_in.shape[3] / p7_in.shape[3]]
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * F.interpolate(p7_in, scale_factor=scale_factor, mode='nearest')))

        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        scale_factor = [p5_in.shape[2] / p6_up.shape[2],
                        p5_in.shape[3] / p6_up.shape[3]]
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * F.interpolate(p6_up, scale_factor=scale_factor, mode='nearest')))

        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        scale_factor = [p4_in.shape[2] / p5_up.shape[2],
                        p4_in.shape[3] / p5_up.shape[3]]
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * F.interpolate(p5_up, scale_factor=scale_factor, mode='nearest')))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        scale_factor = [p3_in.shape[2] / p4_up.shape[2],
                        p3_in.shape[3] / p4_up.shape[3]]
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * F.interpolate(p4_up, scale_factor=scale_factor, mode='nearest')))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


# class BIFPN(_BIFPN):

#     def forward(self, inputs):
#         assert len(inputs) == len(self.in_channels)

#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i + self.start_level])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]

#         # part 1: build top-down and down-top path with stack
#         used_backbone_levels = len(laterals)
#         for bifpn_module in self.stack_bifpn_convs:
#             laterals = bifpn_module(laterals)
#         outs = laterals
#         # part 2: add extra levels
#         if self.num_outs > len(outs):
#             # use max pool to get more levels on top of outputs
#             # (e.g., Faster R-CNN, Mask R-CNN)
#             if not self.add_extra_convs:
#                 for i in range(self.num_outs - used_backbone_levels):
#                     outs.append(F.max_pool2d(outs[-1], 1, stride=2))
#             # add conv layers on top of original feature maps (RetinaNet)
#             else:
#                 if self.extra_convs_on_inputs:
#                     orig = inputs[self.backbone_end_level - 1]
#                     outs.append(self.fpn_convs[0](orig))
#                 else:
#                     outs.append(self.fpn_convs[0](outs[-1]))
#                 for i in range(1, self.num_outs - used_backbone_levels):
#                     if self.relu_before_extra_convs:
#                         outs.append(self.fpn_convs[i](F.relu(outs[-1])))
#                     else:
#                         outs.append(self.fpn_convs[i](outs[-1]))

#         names = [0, 1, 2, 3, 'pool']
#         return OrderedDict([(n, o) for n, o in zip(names, outs)])


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        # initial_shape = x.shape
        res = []
        for conv in self.convs:
            res.append(conv(x))
            # try:
            #     res.append(conv(x))
            # except RuntimeError:
            #     print(initial_shape, x.shape)
            #     raise
        res = torch.cat(res, dim=1)
        return self.project(res)


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, gt_proposal_ids):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss


def maskrcnn_loss_lovasz(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, gt_proposal_ids):
    """
    Arguments:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    # if len(gt_masks) == 2:
    #     print([torch.unique(m) for m in gt_masks])
    #     print([(torch.max(m), m.shape) for m in mask_targets[:-1]])

    mask_targets[mask_targets > 1.0] = 255.0
    # print([torch.max(m) for m in mask_targets])

    mask_loss = lovasz_hinge(mask_logits[torch.arange(labels.shape[0], device=labels.device), labels],
                             mask_targets, ignore=255.0)

    return mask_loss


def roi_heads_forward(self, features, proposals, image_shapes, targets=None, box_coord_perm=None, images=None):
    """
    Arguments:
        features (List[Tensor])
        proposals (List[Tensor[N, 4]])
        image_shapes (List[Tuple[H, W]])
        targets (List[Dict])
    """
    if targets is not None:
        for t in targets:
            assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
            assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
            if self.has_keypoint:
                assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

    if self.training:
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    result, losses = [], {}
    if self.training:
        loss_classifier, loss_box_reg = fastrcnn_loss(
            class_logits, box_regression, labels, regression_targets, box_coord_perm)
        losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
    else:
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                dict(
                    boxes=boxes[i],
                    labels=labels[i],
                    scores=scores[i],
                )
            )

    if self.has_mask:
        mask_proposals = [p["boxes"] for p in result]
        if self.training:
            # during training, only focus on positive boxes
            num_images = len(proposals)
            mask_proposals = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                mask_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])


        mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)

        if mask_features.shape[0] > 0:
            mask_features = self.mask_head(mask_features)
        elif len(mask_proposals) > 1:
            raise NotImplementedError
        mask_logits = self.mask_predictor(mask_features)

        loss_mask = {}
        gt_proposal_ids = None
        if self.training:
            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]

            if self.maskrcnn_loss == 'BCE':
                maskrcnn_loss_func = maskrcnn_loss
            elif self.maskrcnn_loss == 'LOVASZ':
                maskrcnn_loss_func = maskrcnn_loss_lovasz
            else:
                raise NotImplementedError

            loss_mask = maskrcnn_loss_func(
                mask_logits, mask_proposals,
                gt_masks, gt_labels, pos_matched_idxs,
                gt_proposal_ids)

            loss_mask = dict(loss_mask=loss_mask)
        else:
            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)

            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        losses.update(loss_mask)

    if self.has_keypoint:
        keypoint_proposals = [p["boxes"] for p in result]
        if self.training:
            # during training, only focus on positive boxes
            num_images = len(proposals)
            keypoint_proposals = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                keypoint_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])

        keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
        keypoint_features = self.keypoint_head(keypoint_features)
        keypoint_logits = self.keypoint_predictor(keypoint_features)

        loss_keypoint = {}
        if self.training:
            gt_keypoints = [t["keypoints"] for t in targets]
            loss_keypoint = keypointrcnn_loss(
                keypoint_logits, keypoint_proposals,
                gt_keypoints, pos_matched_idxs)
            loss_keypoint = dict(loss_keypoint=loss_keypoint)
        else:
            keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
            for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                r["keypoints"] = keypoint_prob
                r["keypoints_scores"] = kps

        losses.update(loss_keypoint)

    return result, losses


def rpn_forward(self, images, features, targets=None):
    """
    Arguments:
        images (ImageList): images for which we want to compute the predictions
        features (List[Tensor]): features computed from the images that are
            used for computing the predictions. Each tensor in the list
            correspond to different feature levels
        targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
            If provided, each element in the dict should contain a field `boxes`,
            with the locations of the ground-truth boxes.

    Returns:
        boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
            image.
        losses (Dict[Tensor]): the losses for the model during training. During
            testing, it is an empty dict.
    """

    # RPN uses all feature maps that are available
    features = list(features.values())
    objectness, pred_bbox_deltas = self.head(features)
    anchors = self.anchor_generator(images, features)

    num_images = len(anchors)
    num_anchors_per_level = [o[0].numel() for o in objectness]
    objectness, pred_bbox_deltas = \
        concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    if not self.training and targets is not None and self._eval_augment_proposals_mode is not None:
    # if targets is not None and self._eval_augment_proposals_mode is not None:
        # boxes = []
        random_share = 0.1
        num_box_augs = self.post_nms_top_n
        if self._eval_augment_proposals_mode == 'EXTEND':
            num_box_augs = self.post_nms_top_n // 2

        for i, target in enumerate(targets):
            img = images.tensors[i]
            img_height, img_width = img.shape[-2:]

            target_boxes = []
            # for box in boxes_width_height:
            for box in target['boxes']:
                # if self.augment_target_proposals:
                box_width, box_height = box[2] - box[0], box[3] - box[1]

                # if self._eval_augment_proposals_mode == 'EXTEND':
                random_x_mins = box[0] - torch.rand((num_box_augs,)) * box_width * random_share
                random_y_mins = box[1] - torch.rand((num_box_augs,)) * box_height * random_share
                random_x_maxs = box[2] + torch.rand((num_box_augs,)) * box_width * random_share
                random_y_maxs = box[3] + torch.rand((num_box_augs,)) * box_height * random_share
                # elif self._eval_augment_proposals_mode == 'REPLACE':
                #     random_x_mins = box[0] - torch.ones((1,)) * box_width * 0.2
                #     random_y_mins = box[1] - torch.ones((1,)) * box_height * 0.2
                #     random_x_maxs = box[2] + torch.ones((1,)) * box_width * 0.2
                #     random_y_maxs = box[3] + torch.ones((1,)) * box_height * 0.2

                box_augs = torch.stack([random_x_mins.clamp(0, img_width),
                                        random_y_mins.clamp(0, img_height),
                                        random_x_maxs.clamp(0, img_width),
                                        random_y_maxs.clamp(0, img_height)], dim=1)

                # import numpy as np
                # import matplotlib.pyplot as plt
                # fig = plt.figure()
                # ax = plt.Axes(fig, [0., 0., 1., 1.])
                # ax.set_axis_off()
                # fig.add_axes(ax)
                #
                # # ax.imshow(images[i], cmap='jet', vmin=0, vmax=test_loader.dataset.num_objects)
                # ax.imshow(np.transpose(images.tensors[i].mul(255).cpu().numpy(), (1, 2, 0)).astype(np.uint8))
                #
                # ax.add_patch(
                #     plt.Rectangle(
                #         (box[0].item(), box[1].item()),
                #         box[2].item() - box[0].item(),
                #         box[3].item() - box[1].item(),
                #         fill=False,
                #         linewidth=2.0,
                #     ))
                # for box_a in box_augs:
                #     ax.add_patch(
                #         plt.Rectangle(
                #             (box_a[0].item(), box_a[1].item()),
                #             box_a[2].item() - box_a[0].item(),
                #             box_a[3].item() - box_a[1].item(),
                #             fill=False,
                #             linewidth=1.0,
                #         ))
                #
                # plt.axis('off')
                # # plt.tight_layout()
                # plt.draw()
                # plt.savefig('box_augs.png', dpi=100)
                # plt.close()
                # exit()

                target_boxes.append(box_augs)

            target_boxes = torch.cat(target_boxes, dim=0)
            target_boxes = target_boxes.to(scores[0].device)

            if self._eval_augment_proposals_mode == 'EXTEND':
                boxes[i] = torch.cat([boxes[i][:self.post_nms_top_n // 2], target_boxes], dim=0)
                # boxes[i] = torch.cat([boxes[i][-self.post_nms_top_n // 2:], target_boxes], dim=0)
                # boxes[i] = torch.cat([boxes[i], target_boxes], dim=0)
            elif self._eval_augment_proposals_mode == 'REPLACE':
                boxes[i] = target_boxes
            else:
                raise NotImplementedError

    losses = {}
    if self.training:
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets)
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }
    return boxes, losses


def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    # split boxes and scores per image
    pred_boxes = pred_boxes.split(boxes_per_image, 0)
    pred_scores = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        # print(self.training, boxes.shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.flatten()
        labels = labels.flatten()

        orig_inds = torch.arange(boxes.shape[0])

        # remove low scoring boxes
        inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
        boxes, scores, labels, orig_inds = boxes[inds], scores[inds], labels[inds], orig_inds[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels, orig_inds = boxes[keep], scores[keep], labels[keep], orig_inds[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions

        # if self._eval_augment_proposals_mode == 'EXTEND':
        #     assert self.detections_per_img == 1

        #     use_local = False
        #     for i, orig_ind in enumerate(orig_inds[keep]):
        #         if orig_ind >= 500:
        #             print(orig_ind)
        #             keep = keep[i:i + self.detections_per_img]
        #             use_local = True
        #             break

        #     if not use_local:
        #         print('not local')
        #         keep = keep[:self.detections_per_img]
        # else:
        #    keep = keep[:self.detections_per_img]

        keep = keep[:self.detections_per_img]

        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


class MaskRCNN(_MaskRCNN):

    def __init__(self, backbone, num_classes, batch_norm=None, train_encoder=True,
                 roi_pool_output_sizes=None, eval_augment_rpn_proposals_mode=None,
                 replace_batch_with_group_norms=False, box_nms_thresh=0.5,
                 maskrcnn_loss='LOVASZ'):

        self._num_groups = 32
        if 'efficientnet' in backbone:
            self._num_groups = 8

            compound_coef = int(backbone[-1])

            backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
            fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
            fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
            # fpn_cell_repeats = [2, 2, 2, 2, 2, 2, 2, 2]
            # conv_channel_coef = {
            #     # the channels of P3/P4/P5.
            #     0: [40, 112, 320],
            #     1: [40, 112, 320],
            #     2: [48, 120, 352],
            #     3: [48, 136, 384],
            #     4: [56, 160, 448],
            #     5: [64, 176, 512],
            #     6: [72, 200, 576],
            #     7: [72, 200, 576],
            # }

            conv_channel_coef = {
                # the channels of P3/P4/P5/P6.
                0: [40, 112, 320],
                1: [40, 112, 320],
                2: [48, 120, 352],
                3: [48, 136, 384],
                4: [56, 160, 448],
                5: [64, 176, 512],
                6: [72, 200, 576],
                7: [72, 200, 576],
            }

            backbone_model_encoder = EfficientNet(backbone_compound_coef[compound_coef], False)

            backbone_model_fpn = nn.Sequential(
                *[BiFPN(fpn_num_filters[compound_coef],
                        conv_channel_coef[compound_coef],
                        True if _ == 0 else False,
                        attention=True if compound_coef < 6 else False)
                for _ in range(fpn_cell_repeats[compound_coef])])

            model_state_dict = torch.load(f"models/efficientdet-d{compound_coef}.pth")
            backbone_model_encoder.load_state_dict({k.replace('backbone_net.', ''): v for k, v in model_state_dict.items()
                                                    if 'backbone_net' in k})
            backbone_model_fpn.load_state_dict({k.replace('bifpn.', ''): v for k, v in model_state_dict.items()
                                                if 'bifpn' in k})

            backbone_model = EfficientNetAndBiFPN(backbone_model_encoder, backbone_model_fpn)
            backbone_model.out_channels = fpn_num_filters[compound_coef]
        else:
            backbone_model = resnet_fpn_backbone(backbone, True)

        mask_roi_pool, box_roi_pool = None, None
        if roi_pool_output_sizes is not None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=roi_pool_output_sizes['box'],
                sampling_ratio=2)
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=roi_pool_output_sizes['mask'],
                sampling_ratio=2)

        # rpn_pre_nms_top_n_train = 200
        # rpn_pre_nms_top_n_test = 100
        # rpn_post_nms_top_n_train = 1000
        # rpn_post_nms_top_n_test = 1000

        # large to include all proposals
        # box_batch_size_per_image = 10000

        # bbox_reg_weights = [10.0, 10.0, 10.0, 10.0]

        # mask_head = ASPP(256, [12, 24, 36])
        mask_head = None

        super(MaskRCNN, self).__init__(backbone_model,
                                       num_classes,
                                       box_roi_pool=box_roi_pool,
                                       mask_roi_pool=mask_roi_pool,
                                       mask_head=mask_head,
                                       box_nms_thresh=box_nms_thresh,)
                                    #    box_detections_per_img=1,
                                    #    box_nms_thresh=0.95)
                                    #    bbox_reg_weights=bbox_reg_weights)
                                    #    box_batch_size_per_image=box_batch_size_per_image,)
                                    #    rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                                    #    rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                                    #    rpn_post_nms_top_n_train=rpn_post_nms_top_n_train)
                                    #    rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,)
                                    #    box_positive_fraction=0.25)

        self.num_classes = num_classes
        self.rpn._eval_augment_proposals_mode = eval_augment_rpn_proposals_mode
        self.rpn.forward = types.MethodType(rpn_forward, self.rpn)

        self.roi_heads._eval_augment_proposals_mode = eval_augment_rpn_proposals_mode
        self.roi_heads.postprocess_detections = types.MethodType(
            postprocess_detections, self.roi_heads)

        self.roi_heads.maskrcnn_loss = maskrcnn_loss
        self.roi_heads.forward = types.MethodType(
            roi_heads_forward, self.roi_heads)

        pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
                                                  progress=True)
        state_dict = self.state_dict()

        if 'resnet50' in backbone:

            for k in state_dict.keys():
                if k in pretrained_state_dict and state_dict[k].shape == pretrained_state_dict[k].shape:
                    state_dict[k] = pretrained_state_dict[k]

            self.load_state_dict(state_dict)
        elif 'efficientnet' in backbone:
            for k in state_dict.keys():
                if 'backbone' not in k and k in pretrained_state_dict and state_dict[k].shape == pretrained_state_dict[k].shape:
                    state_dict[k] = pretrained_state_dict[k]
            self.load_state_dict(state_dict)

        if replace_batch_with_group_norms:
            self.replace_batch_with_group_norms()

        self._train_encoder = train_encoder
        if not train_encoder:
            self.requires_grad_(False)
            # self.backbone.body.layer4.requires_grad_(True)
            # self.backbone.fpn.requires_grad_(True)
            # self.backbone.requires_grad_(True)

            self.rpn.requires_grad_(True)

            self.roi_heads.box_head.requires_grad_(True)
            self.roi_heads.box_predictor.requires_grad_(True)
            self.roi_heads.mask_head.requires_grad_(True)
            self.roi_heads.mask_predictor.requires_grad_(True)
        else:
            self.backbone.requires_grad_(True)
            # self.backbone.body.conv1.requires_grad_(False)
            # self.backbone.body.layer1.requires_grad_(False)

            # self.rpn.requires_grad_(False)

        #     self.backbone.fpn.requires_grad_(True)
        #     self.rpn.requires_grad_(True)

        self._accum_batch_norm_stats = True
        if batch_norm is not None:
            self._accum_batch_norm_stats = batch_norm['accum_stats']

            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.requires_grad = batch_norm['learn_weight']
                    m.bias.requires_grad = batch_norm['learn_bias']

        # self._second_order_derivates_module_names = ['rpn', 'roi_heads']
        self._second_order_derivates_module_names = ['roi_heads']

        self.last_param_group_names = ['roi_heads.box_predictor.cls_score.weight',
                                       'roi_heads.box_predictor.cls_score.bias',
                                       'roi_heads.box_predictor.bbox_pred.weight',
                                       'roi_heads.box_predictor.bbox_pred.bias',
                                       'roi_heads.mask_predictor.mask_fcn_logits.weight',
                                       'roi_heads.mask_predictor.mask_fcn_logits.bias']


    def replace_batch_with_group_norms(self):
        for module in self.modules():
            bn_keys = [k for k, m in module._modules.items()
                       if isinstance(m, FrozenBatchNorm2d) or isinstance(m, nn.BatchNorm2d)]
            for k in bn_keys:
                batch_norm = module._modules[k]

                group_norm = nn.GroupNorm(self._num_groups, batch_norm.weight.shape[0])
                group_norm.weight.data = batch_norm.weight
                group_norm.bias.data = batch_norm.bias

                module._modules[k] = group_norm

    def named_parameters_with_second_order_derivate(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad and any([module_name in name for module_name in self._second_order_derivates_module_names]):
                yield name, param

    def named_parameters_without_second_order_derivate(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad and not any([module_name in name for module_name in self._second_order_derivates_module_names]):
                yield name, param

    def train(self, mode=True):
        super(MaskRCNN, self).train(mode)
        if not self._train_encoder:
            self.backbone.eval()
            # self.backbone.body.layer4.train()
            # self.backbone.fpn.train()
            # self.rpn.eval()
        # else:
        #     self.backbone.body.layer1.eval()
        #     self.backbone.body.conv1.eval()
        #     # self.backbone.eval()
        #     self.rpn.eval()
        #     # self.backbone.fpn.train()

        if not self._accum_batch_norm_stats:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

    def train_without_dropout(self):
        self.train()

        for m in self.modules():
            if isinstance(m, torch.nn.Dropout2d) or isinstance(m, torch.nn.Dropout):
                m.eval()

    def forward(self, inputs, targets=None, box_coord_perm=None, flip_label=False):

        # # TODO: solve not here.
        # _, _, h, w = inputs.shape
        # pad = [0, 0, 0, 0]
        # crop = [0, 0, 0, 0]

        # inputs_padded = F.pad(input=inputs, pad=pad, mode='constant', value=0)
        device = inputs.device

        if targets is not None:
            # assert self.training

            if flip_label:
                targets  = 1 - targets

            mask_rcnn_targets = []
            for target in targets:

                mask = target
                # instances are encoded as different colors
                obj_ids = torch.unique(mask)

                # first id is the background, so remove it
                obj_ids = torch.tensor([obj_id.item() for obj_id in obj_ids
                                        if obj_id.item() != 0.0 and obj_id.item() != 255.0]).to(obj_ids.device)
                # obj_ids = obj_ids[1:]

                # split the color-encoded mask into a set
                # of binary masks
                masks = mask == obj_ids[:, None, None]

                masks[mask == 255.0] = True

                # get bounding box coordinates for each mask
                num_objs = len(obj_ids)

                if num_objs == 0:
                    pred = np.transpose(inputs[0].cpu().numpy(), (1, 2, 0))
                    import os, imageio
                    pred_path = os.path.join(f"img.png")
                    imageio.imsave(pred_path, pred)

                    # pred = np.transpose(masks.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                    # # pred = mask.cpu().numpy().astype(np.uint8)
                    # import os, imageio
                    # pred_path = os.path.join(f"mask.png")
                    # imageio.imsave(pred_path, 20 * pred)

                    assert num_objs == 1, f"num_objs: {num_objs}"

                boxes = []
                for i in range(num_objs):
                    pos = np.where(masks[i].cpu().numpy())
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1]) + 1
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0]) + 1
                    boxes.append([xmin, ymin, xmax, ymax])

                # convert everything into a torch.Tensor
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                # there is only one class

                # labels = torch.ones((num_objs,), dtype=torch.int64)
                labels = obj_ids.type(torch.int64)

                masks = masks.type(torch.uint8)
                # masks[mask == 255.0] = 255.0
                if (mask == 255.0).any():
                    masks[mask == 255.0] = 255.0
                    masks[mask == 0.0] = 255.0

                # if self.training:
                #     print(torch.unique(masks))

                image_id = torch.tensor([0])

                # # if not num_objs:
                # pred = np.transpose(masks.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                # # pred = mask.cpu().numpy().astype(np.uint8)
                # import os, imageio
                # pred_path = os.path.join(f"mask.png")
                # imageio.imsave(pred_path, 20 * pred)

                # pred = np.transpose(inputs[0].cpu().numpy(), (1, 2, 0))
                # # pred = mask.cpu().numpy().astype(np.uint8)
                # import os, imageio
                # pred_path = os.path.join(f"img.png")
                # imageio.imsave(pred_path, (pred * 255).astype(np.uint8))


                # # pred_ = np.transpose(preds[frame_id].cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                # # # pred = mask.cpu().numpy().astype(np.uint8)
                # # imageio.imsave(f"mask.png", 20 * pred_)

                # # pred_ *= 20
                # import matplotlib.pyplot as plt

                # pred = np.transpose(masks.cpu().numpy(),
                #                     (1, 2, 0)).astype(np.uint8)
                # fig = plt.figure()
                # ax = plt.Axes(fig, [0., 0., 1., 1.])
                # ax.set_axis_off()
                # fig.add_axes(ax)
                # ax.imshow(pred.squeeze(2))

                # for box in boxes:
                #     ax.add_patch(
                #         plt.Rectangle(
                #             (box[0], box[1]),
                #             box[2] - box[0],
                #             box[3] - box[1],
                #             fill=False,
                #             linewidth=1.0,
                #         ))

                # plt.axis('off')
                # # plt.tight_layout()
                # plt.draw()
                # plt.savefig(f"mask_with_boxes.png", dpi=100)
                # plt.close()
                # # exit()

                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                # suppose all instances are not crowd
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

                if flip_label:
                    masks = 1 - masks

                mask_rcnn_target = {}
                mask_rcnn_target["boxes"] = boxes.to(device)
                mask_rcnn_target["labels"] = labels.to(device)
                mask_rcnn_target["masks"] = masks.to(device)
                mask_rcnn_target["image_id"] = image_id.to(device)
                mask_rcnn_target["area"] = area.to(device)
                mask_rcnn_target["iscrowd"] = iscrowd.to(device)

                mask_rcnn_targets.append(mask_rcnn_target)
            targets = mask_rcnn_targets

        outputs_raw = super(MaskRCNN, self).forward([i for i in inputs], targets, box_coord_perm)

        # if crop[0]:
        #     outputs = outputs[..., crop[0]:]
        # if crop[1]:
        #     outputs = outputs[..., : -crop[1]]
        # if crop[2]:
        #     outputs = outputs[:,:, crop[2]:]
        # if crop[3]:
        #     outputs = outputs[:,:, : -crop[3]]

        if self.training:
            losses = {loss_name: loss for loss_name, loss in outputs_raw.items()
                      if loss.requires_grad}
            loss = sum([loss for loss in losses.values()])
            return loss, losses
        else:
            boxes = torch.zeros(inputs.size(0), 4)
            outputs = torch.zeros_like(inputs)[:, :1]

            assert len(outputs_raw) == 1

            # print(outputs_raw[0])
            # print(outputs_raw[0]['masks'].shape)
            # print(outputs_raw[0]['boxes'].shape)

            # for i, output in enumerate(outputs_raw):
            #     assert len(torch.unique(output['labels'])) == num_objs

            #     if output['masks'].shape[0] >= 1:
            #         outputs[i] = output['masks'][0]
            #         boxes[i] = output['boxes'][0]

            # return outputs, boxes

            # print(torch.cat([o['masks'] for o in outputs_raw['masks'], dim=0).shape)

            output_masks = []
            output_boxes = []
            for output_raw in outputs_raw:
                output_mask = []
                output_box = []

                for i in range(1, self.num_classes):
                    if len((output_raw['labels'] == i).nonzero()):
                        first_index = (output_raw['labels'] == i).nonzero()[0]
                        output_mask.append(output_raw['masks'][first_index][0])
                        output_box.append(output_raw['boxes'][first_index])
                    else:
                        output_mask.append(
                            torch.zeros_like(inputs)[0, 0].unsqueeze(dim=0).to(device))
                        output_box.append(torch.zeros(1, 4).to(device))
                output_masks.append(torch.cat(output_mask, dim=0).unsqueeze(dim=0))
                output_boxes.append(
                    torch.cat(output_box, dim=0).unsqueeze(dim=0))

            # print(torch.cat(output_masks, dim=0).shape)
            # print(torch.cat([o['masks'].unsqueeze(dim=0).squeeze(dim=2) for o in outputs_raw]).shape)

            return torch.cat(output_masks, dim=0), torch.cat(output_boxes, dim=0)
