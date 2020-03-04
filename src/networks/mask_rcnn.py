import numpy as np
import torch
from torchvision.models.detection import MaskRCNN as _MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import MultiScaleRoIAlign


class MaskRCNN(_MaskRCNN):

    def __init__(self, backbone, num_classes, batch_norm=None, train_encoder=True,
                 roi_pool_output_sizes=None):
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
        rpn_post_nms_top_n_test = 2000

        # large to include all proposals
        box_batch_size_per_image = 10000

        # bbox_reg_weights = [10.0, 10.0, 10.0, 10.0]

        super(MaskRCNN, self).__init__(backbone_model,
                                       num_classes,
                                       box_detections_per_img=1,
                                       box_roi_pool=box_roi_pool,
                                       mask_roi_pool=mask_roi_pool,)
                                    #    bbox_reg_weights=bbox_reg_weights)
                                    #    box_batch_size_per_image=box_batch_size_per_image,)
                                    #    rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                                    #    rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                                    #    rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                                    #    rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,)
                                    #    box_positive_fraction=0.25)

        if 'resnet50' == backbone:
            pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
                                                  progress=True)

            state_dict = self.state_dict()

            for k, v in state_dict.items():
                if v.shape != pretrained_state_dict[k].shape:
                    pretrained_state_dict[k] = v

            self.load_state_dict(pretrained_state_dict)

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

            # self.rpn.requires_grad_(True)

            self.roi_heads.box_head.requires_grad_(True)
            self.roi_heads.box_predictor.requires_grad_(True)
            self.roi_heads.mask_head.requires_grad_(True)
            self.roi_heads.mask_predictor.requires_grad_(True)
        # else:
        #     self.backbone.requires_grad_(True)

        #     self.backbone.fpn.requires_grad_(True)
        #     self.rpn.requires_grad_(True)

        # self._second_order_derivates_module_names = ['rpn', 'roi_heads']
        self._second_order_derivates_module_names = ['roi_heads']

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
            self.rpn.eval()
        # else:
        #     self.backbone.eval()
        #     self.rpn.train()
        #     self.backbone.fpn.train()

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
                # pred = mask.cpu().numpy().astype(np.uint8)
                # import os, imageio
                # pred_path = os.path.join(f"img.png")
                # imageio.imsave(pred_path, (pred * 255).astype(np.uint8))

                # if frame_id == 7:

                # pred_ = np.transpose(preds[frame_id].cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                # # pred = mask.cpu().numpy().astype(np.uint8)
                # imageio.imsave(f"mask.png", 20 * pred_)

                # pred_ *= 20
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
                # exit()

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
