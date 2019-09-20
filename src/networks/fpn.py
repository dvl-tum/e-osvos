import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


class FPN(smp.FPN):

    def forward(self, inputs):
        b, c, h, w = inputs.shape

        # TODO: solve nicer
        if w == 854:
            pad = (4, 5, 0, 0)
            crop = 5
        elif w == 910:
            pad = (7, 8, 0, 0)
            crop = 9
        elif w == 1152:
            pad = (0, 0, 0, 0)
            crop = 0
        else:
            raise NotImplementedError
        
        inputs = F.pad(input=inputs, pad=pad, mode='constant', value=0)
        outputs = super(FPN, self).forward(inputs)
        if crop:
            return [outputs[..., crop: -crop]]
        else:
            return [outputs]

    def train_no_batch_norm(self, mode=True):
        super(FPN, self).train(mode)

        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    def modules_with_requires_grad_params(self):
        # _parameters includes only direct parameters of a module and not all
        # parameters of its potential submodules.
        for module in self.modules():
            if len(module._parameters):
                for p in module._parameters.values():
                    if p is not None and p.requires_grad:
                        yield module
                        break
