import segmentation_models_pytorch as smp
import torch.nn.functional as F


class FPN(smp.FPN):

    def forward(self, inputs):
        inputs = F.pad(input=inputs, pad=(4, 5, 0, 0), mode='constant', value=0)
        outputs = super(FPN, self).forward(inputs)
        return [outputs[..., 5:-5]]

        # outputs = super(FPN, self).forward(inputs)
        # return [outputs[..., 1:-1, 1:-1]]
