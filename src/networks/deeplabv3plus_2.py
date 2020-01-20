import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabV3 as _DeepLabV3
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
import torch.nn.functional as F


import math
import torch.nn as nn
from collections import OrderedDict


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            # ASPP(in_channels, [6, 12, 18]),
        )

class _DeepLabV3Plus2(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, decoder, aux_classifier=None):
        super(_DeepLabV3Plus2, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.decoder = decoder
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = self.decoder(x, features["low_level_feat"])
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape,
                              mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        low_level_inplanes = 256
        # if backbone == 'resnet' or backbone == 'drn':
        #     low_level_inplanes = 256
        # elif backbone == 'xception':
        #     low_level_inplanes = 128
        # elif backbone == 'mobilenet':
        #     low_level_inplanes = 24
        # else:
        #     raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3,
                                                 stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[
                          2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabV3Plus2(_DeepLabV3Plus2):

    def __init__(self, backbone, num_classes, batch_norm=None, train_encoder=True):
        classifier = DeepLabHead(2048, num_classes)
        aux_classifier = None
        return_layers = {'layer4': 'out', 'layer1': 'low_level_feat'}
        # if aux:
        #     return_layers['layer3'] = 'aux'
        backbone_model = resnet.__dict__[backbone](
            pretrained=True,
            replace_stride_with_dilation=[False, True, True])
        backbone_model = IntermediateLayerGetter(backbone_model, return_layers=return_layers)
        decoder = Decoder(num_classes)
        super(DeepLabV3Plus2, self).__init__(backbone_model, classifier, decoder, aux_classifier)

        if 'resnet101' == backbone:
            pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
                                                             progress=True)
            state_dict = self.state_dict()

            del_keys = [k for k in pretrained_state_dict.keys() if k not in state_dict]
            for k in del_keys:
                del pretrained_state_dict[k]

            for k, v in state_dict.items():
                if k not in pretrained_state_dict:
                    pretrained_state_dict[k] = v

            self.load_state_dict(pretrained_state_dict)

        if not train_encoder:
            self.backbone.requires_grad_(False)
            self.backbone.layer4.requires_grad_(True)

        self._accum_batch_norm_stats = True
        if batch_norm is not None:
            self._accum_batch_norm_stats = batch_norm['accum_stats']

            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.requires_grad = batch_norm['learn_weight']
                    m.bias.requires_grad = batch_norm['learn_bias']

        # print('backbone.layer4 ', sum([p.numel() for p in self.backbone.layer4.parameters() if p.requires_grad]))
        # print('classifier ', sum([p.numel() for p in self.classifier.parameters() if p.requires_grad]))
        # print('decoder ', sum([p.numel() for p in self.decoder.parameters() if p.requires_grad]))

    def train(self, mode=True):
        super(DeepLabV3Plus2, self).train(mode)

        if not self._accum_batch_norm_stats:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

    def train_without_dropout(self):
        self.train()

        for m in self.modules():
            if isinstance(m, torch.nn.Dropout2d) or isinstance(m, torch.nn.Dropout):
                m.eval()

    def forward(self, inputs):

        # TODO: solve not here.
        _, _, h, w = inputs.shape
        pad = [0, 0, 0, 0]
        crop = [0, 0, 0, 0]

        inputs_padded = F.pad(input=inputs, pad=pad, mode='constant', value=0)
        outputs = super(DeepLabV3Plus2, self).forward(inputs_padded)['out']

        if crop[0]:
            outputs = outputs[..., crop[0]:]
        if crop[1]:
            outputs = outputs[..., : -crop[1]]
        if crop[2]:
            outputs = outputs[:,:, crop[2]:]
        if crop[3]:
            outputs = outputs[:,:, : -crop[3]]

        return [outputs]
