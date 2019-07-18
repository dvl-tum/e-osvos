import segmentation_models_pytorch as smp
import torch.nn.functional as F


class Unet(smp.Unet):

    def forward(self, inputs):
        inputs = F.pad(input=inputs, pad=(4, 5, 0, 0),
                       mode='constant', value=0)
        outputs = super(Unet, self).forward(inputs)
        return [outputs[..., 5:-5]]
