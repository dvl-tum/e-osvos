import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


class FPN(smp.FPN):

    def __init__(self, *args, batch_norm=None, **kwargs):
        super(FPN, self).__init__(*args, **kwargs)

        self._accum_batch_norm_stats = True
        if batch_norm is not None:
            self._accum_batch_norm_stats = batch_norm['accum_stats']

            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.requires_grad = batch_norm['learn_weight']
                    m.bias.requires_grad = batch_norm['learn_bias']

    def forward(self, inputs):
        # TODO: solve not here.
        _, _, h, w = inputs.shape
        pad = [0, 0, 0, 0]
        crop = [0, 0, 0, 0]
        if w == 854:
            pad[0] = 4
            pad[1] = 5
            crop[0] = 5
            crop[1] = 5
        elif w == 910:
            pad[0] = 7
            pad[1] = 8
            crop[0] = 9
            crop[1] = 9
        elif w == 911:
            pad[0] = 7
            pad[1] = 7
            crop[0] = 8
            crop[1] = 9
        elif w == 1138:
            pad[0] = 5
            pad[1] = 6
            crop[0] = 7
            crop[1] = 7

        if h == 720:
            pad[2] = 6
            pad[3] = 7
            crop[2] = 8
            crop[3] = 8

        inputs = F.pad(input=inputs, pad=pad, mode='constant', value=0)
        outputs = super(FPN, self).forward(inputs)

        if crop[0]:
            outputs = outputs[..., crop[0]:]
        if crop[1]:
            outputs = outputs[..., : -crop[1]]
        if crop[2]:
            outputs = outputs[:,:, crop[2]:]
        if crop[3]:
            outputs = outputs[:,:, : -crop[3]]

        return [outputs]

    def train(self, mode=True):
        super(FPN, self).train(mode)

        if not self._accum_batch_norm_stats:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

    def modules_with_requires_grad_params(self):
        # _parameters includes only direct parameters of a module and not all
        # parameters of its potential submodules.
        # for module in self.modules():
        #     if len(module._parameters):
        #         for p in module._parameters.values():
        #             if p is not None and p.requires_grad:
        #                 yield module
        #                 break

        for n, m in self.named_modules():
            if n == 'encoder':
                yield m
