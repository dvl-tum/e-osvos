import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import MaskRCNN as _MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d


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


class MaskRCNN(_MaskRCNN):

    def __init__(self, backbone, num_classes, batch_norm=None, train_encoder=True,
                 roi_pool_output_sizes=None, eval_augment_rpn_proposals_mode=None,
                 replace_batch_with_group_norms=False):
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
        # rpn_post_nms_top_n_train = 200
        # rpn_post_nms_top_n_test = 2000

        # large to include all proposals
        # box_batch_size_per_image = 10000

        # bbox_reg_weights = [10.0, 10.0, 10.0, 10.0]

        # mask_head = ASPP(256, [12, 24, 36])
        mask_head = None

        super(MaskRCNN, self).__init__(backbone_model,
                                       num_classes,
                                       box_detections_per_img=1,
                                       box_roi_pool=box_roi_pool,
                                       mask_roi_pool=mask_roi_pool,
                                       mask_head=mask_head)
                                    #    bbox_reg_weights=bbox_reg_weights)
                                    #    box_batch_size_per_image=box_batch_size_per_image,)
                                    #    rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                                    #    rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                                    #    rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                                    #    rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,)
                                    #    box_positive_fraction=0.25)

        self.num_classes = num_classes
        self.rpn._eval_augment_proposals_mode = eval_augment_rpn_proposals_mode
        self.rpn.forward = types.MethodType(rpn_forward, self.rpn)

        if 'resnet50' == backbone:
            pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
                                                  progress=True)

            state_dict = self.state_dict()

            for k in state_dict.keys():
                if k in pretrained_state_dict and state_dict[k].shape == pretrained_state_dict[k].shape:
                    state_dict[k] = pretrained_state_dict[k]

            self.load_state_dict(state_dict)

        self._accum_batch_norm_stats = True
        if batch_norm is not None:
            self._accum_batch_norm_stats = batch_norm['accum_stats']

            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.requires_grad = batch_norm['learn_weight']
                    m.bias.requires_grad = batch_norm['learn_bias']

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
            self.backbone.body.conv1.requires_grad_(False)
            self.backbone.body.layer1.requires_grad_(False)

            # self.rpn.requires_grad_(False)

        #     self.backbone.fpn.requires_grad_(True)
        #     self.rpn.requires_grad_(True)

        # self._second_order_derivates_module_names = ['rpn', 'roi_heads']
        self._second_order_derivates_module_names = ['roi_heads']

        if replace_batch_with_group_norms:
            self.replace_batch_with_group_norms()

        self.last_param_group_names = ['roi_heads.box_predictor.cls_score.weight',
                                       'roi_heads.box_predictor.cls_score.bias',
                                       'roi_heads.box_predictor.bbox_pred.weight',
                                       'roi_heads.box_predictor.bbox_pred.bias',
                                       'roi_heads.mask_predictor.mask_fcn_logits.weight',
                                       'roi_heads.mask_predictor.mask_fcn_logits.bias']

    def replace_batch_with_group_norms(self):
        for module in self.modules():
            bn_keys = [k for k, m in module._modules.items()
                       if isinstance(m, FrozenBatchNorm2d)]
            for k in bn_keys:
                batch_norm = module._modules[k]

                group_norm = nn.GroupNorm(32, batch_norm.weight.shape[0])
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
        else:
            self.backbone.body.layer1.eval()
            self.backbone.body.conv1.eval()
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
                                        if obj_id.item() != 0.0]).to(obj_ids.device)
                # obj_ids = obj_ids[1:]

                # split the color-encoded mask into a set
                # of binary masks
                masks = mask == obj_ids[:, None, None]

                # get bounding box coordinates for each mask
                num_objs = len(obj_ids)

                if num_objs != 1:
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

                labels = torch.ones((num_objs,), dtype=torch.int64)
                # masks = torch.as_tensor(masks, dtype=torch.uint8)
                masks = masks.type(torch.uint8)

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
            for i, output in enumerate(outputs_raw):
                if output['masks'].shape[0] >= 1:
                    outputs[i] = output['masks'][0]
                    boxes[i] = output['boxes'][0]

            return outputs, boxes
