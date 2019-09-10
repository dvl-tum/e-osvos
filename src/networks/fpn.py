import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


class FPN(smp.FPN):

    def forward(self, inputs):
        inputs = F.pad(input=inputs, pad=(4, 5, 0, 0), mode='constant', value=0)
        outputs = super(FPN, self).forward(inputs)
        return [outputs[..., 5:-5]]

        # outputs = super(FPN, self).forward(inputs)
        # return [outputs[..., 1:-1, 1:-1]]

    def train(self, mode=True):
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
