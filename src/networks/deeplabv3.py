import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabV3 as _DeepLabV3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
import torch.nn.functional as F


class DeepLabV3(_DeepLabV3):

    def __init__(self, backbone, num_classes, batch_norm=None, train_encoder=True):
        classifier = DeepLabHead(2048, num_classes)
        aux_classifier = None
        return_layers = {'layer4': 'out'}
        # if aux:
        #     return_layers['layer3'] = 'aux'
        backbone_model = resnet.__dict__[backbone](
            pretrained=True,
            replace_stride_with_dilation=[False, True, True])
        backbone_model = IntermediateLayerGetter(backbone_model, return_layers=return_layers)
        super(DeepLabV3, self).__init__(backbone_model, classifier, aux_classifier)

        if 'resnet101' == backbone:
            pretrained_state_dict = load_state_dict_from_url('https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
                                                             progress=True)
            state_dict = self.state_dict()

            pretrained_state_dict['classifier.4.weight'] = state_dict['classifier.4.weight']
            pretrained_state_dict['classifier.4.bias'] = state_dict['classifier.4.bias']
            # pretrained_state_dict['aux_classifier.4.weight'] = state_dict['aux_classifier.4.weight']
            # pretrained_state_dict['aux_classifier.4.bias'] = state_dict['aux_classifier.4.bias']
            del pretrained_state_dict['aux_classifier.4.weight']
            del pretrained_state_dict['aux_classifier.4.bias']
            del pretrained_state_dict["aux_classifier.0.weight"]
            del pretrained_state_dict["aux_classifier.1.weight"]
            del pretrained_state_dict["aux_classifier.1.bias"]
            del pretrained_state_dict["aux_classifier.1.running_mean"]
            del pretrained_state_dict["aux_classifier.1.running_var"]
            del pretrained_state_dict["aux_classifier.1.num_batches_tracked"]

            self.load_state_dict(pretrained_state_dict)

        self._accum_batch_norm_stats = True
        if batch_norm is not None:
            self._accum_batch_norm_stats = batch_norm['accum_stats']

            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.requires_grad = batch_norm['learn_weight']
                    m.bias.requires_grad = batch_norm['learn_bias']

        if not train_encoder:
            self.backbone.requires_grad_(False)

    def train(self, mode=True):
        super(DeepLabV3, self).train(mode)

        if not self._accum_batch_norm_stats:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

    def forward(self, inputs):

        # TODO: solve not here.
        _, _, h, w = inputs.shape
        pad = [0, 0, 0, 0]
        crop = [0, 0, 0, 0]

        inputs_padded = F.pad(input=inputs, pad=pad, mode='constant', value=0)
        outputs = super(DeepLabV3, self).forward(inputs_padded)['out']

        if crop[0]:
            outputs = outputs[..., crop[0]:]
        if crop[1]:
            outputs = outputs[..., : -crop[1]]
        if crop[2]:
            outputs = outputs[:,:, crop[2]:]
        if crop[3]:
            outputs = outputs[:,:, : -crop[3]]

        return [outputs]