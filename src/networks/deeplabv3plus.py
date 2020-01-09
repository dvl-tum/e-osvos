import torch
import torch.nn.functional as F


from pytorch_deeplab_xception.modeling.deeplab import DeepLab


class DeepLabV3Plus(DeepLab):

    def __init__(self, *args, batch_norm=None, train_encoder=True, **kwargs):
        super(DeepLabV3Plus, self).__init__(*args, **kwargs)

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
        super(DeepLabV3Plus, self).train(mode)

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
        outputs = super(DeepLabV3Plus, self).forward(inputs_padded)

        if crop[0]:
            outputs = outputs[..., crop[0]:]
        if crop[1]:
            outputs = outputs[..., : -crop[1]]
        if crop[2]:
            outputs = outputs[:, :, crop[2]:]
        if crop[3]:
            outputs = outputs[:, :, : -crop[3]]

        return [outputs]
