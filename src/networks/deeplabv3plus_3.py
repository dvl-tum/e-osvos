import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict

affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual
        out = self.relu(out)

        return out


class Deeplabv3PlusDecoderHead(nn.Module):

    def __init__(self, dilations, num_classes):
        super(Deeplabv3PlusDecoderHead, self).__init__()

        self.assp = []
        self.assp = nn.ModuleList(
            [self._make_assp(2048, 256, dilation) for dilation in dilations])
        self.image_pool = self._make_image_fea(2048, 256)

        self.encode_fea = self._make_block(256*5, 256, ks=1, pad=0)
        self.low_level_fea = self._make_block(256, 48, ks=1, pad=0)
        self.decode_1 = self._make_block(256+48, 256, ks=3, pad=1)
        self.decode_2 = self._make_block(256, 256, ks=3, pad=1)
        self.final = nn.Conv2d(
            256, num_classes, kernel_size=1, stride=1, bias=True)

    def _make_assp(self, in_features, out_features, dilation):
        if dilation == 1:
            conv = nn.Conv2d(in_features, out_features,
                             kernel_size=1, stride=1, bias=False)
        else:
            conv = nn.Conv2d(in_features, out_features, kernel_size=3,
                             stride=1, padding=dilation, dilation=dilation, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def _make_image_fea(self, in_features, out_features):
        g_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        conv = nn.Conv2d(in_features, out_features,
                         kernel_size=1, stride=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(g_pool, conv, bn, relu)

    def _make_block(self, in_features, out_features, ks=1, pad=0):
        conv = nn.Conv2d(in_features, out_features,
                         kernel_size=ks, padding=pad, stride=1, bias=False)
        bn = nn.BatchNorm2d(out_features, affine=affine_par)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)

    def forward(self, res2_fea, res5_fea):
        # get encoder feature
        h, w = res5_fea.size(2), res5_fea.size(3)
        image_fea = F.upsample(input=self.image_pool(
            res5_fea), size=(h, w), mode='bilinear')
        encoder_fea = [assp_stage(res5_fea) for assp_stage in self.assp]
        encoder_fea.append(image_fea)
        encoder_fea = self.encode_fea(torch.cat(encoder_fea, dim=1))

        # get low level feature
        low_level_fea = self.low_level_fea(res2_fea)

        # concat and decode
        h, w = res2_fea.size(2), res2_fea.size(3)
        encoder_fea = F.upsample(
            input=encoder_fea, size=(h, w), mode='bilinear')
        out = self.decode_1(torch.cat((encoder_fea, low_level_fea), dim=1))
        out = self.decode_2(out)
        out = self.final(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, output_stride):
        self.inplanes = 64
        super(ResNet, self).__init__()

        # setting dilated convolution rates
        if output_stride == 16:
            layer3_stride, layer4_stride = 2, 1
            layer3_dilation = [1]*layers[2]
            layer4_dilation = [2, 4, 8]
        elif output_stride == 8:
            layer3_stride, layer4_stride = 1, 1
            layer3_dilation = [2]*layers[2]
            layer4_dilation = [4, 8, 16]
        else:
            raise ValueError('Undifined output stride rate!')

        # build network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.layer1 = self._make_layer(
            block,  64, layers[0], stride=1, dilation=[1]*layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilation=[1]*layers[1])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=layer3_stride, dilation=layer3_dilation)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=layer4_stride, dilation=layer4_dilation)
        self.layer5 = Deeplabv3PlusDecoderHead((1, 6, 12, 18), num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, dilation):
        downsample = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.inplanes, planes * block.expansion,
                               kernel_size=1, stride=stride, bias=False)),
            ('bn', nn.BatchNorm2d(planes * block.expansion, affine=affine_par))]))

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            dilation=dilation[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation[i]))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, num_classes):
        return block(dilation_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        res2_fea = self.layer1(x)
        x = self.layer2(res2_fea)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.layer5(res2_fea, x)

        return x

    def set_learnable_params(self, layers, bn_names):
        # print ('Set meta learn parameters:', end=' ')
        for name, p in self.named_parameters():
            if any([name.startswith(l) for l in layers]):
                if 'bn' in name or any([k in name for k in bn_names]):
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    # print (name, end='-->')
            else:
                p.requires_grad = False
        # print ('END')

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                params[k] = p
        return params

    def copy_meta_weights(self, meta_weights):
        for k, p in self.named_parameters():
            if p.requires_grad and k in meta_weights.keys():
                p.data = meta_weights[k].data.clone()


class DeepLabV3Plus3(ResNet):

    def __init__(self, num_classes, batch_norm=None, train_encoder=True):
        super(DeepLabV3Plus3, self).__init__(
            Bottleneck, [3, 4, 23, 3], num_classes, 16)

        if not train_encoder:
            self.conv1.requires_grad_(False)
            self.bn1.requires_grad_(False)
            self.relu.requires_grad_(False)
            self.maxpool.requires_grad_(False)

            self.layer1.requires_grad_(False)
            self.layer2.requires_grad_(False)
            self.layer3.requires_grad_(False)

        self._accum_batch_norm_stats = True
        if batch_norm is not None:
            self._accum_batch_norm_stats = batch_norm['accum_stats']

            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.requires_grad = batch_norm['learn_weight']
                    m.bias.requires_grad = batch_norm['learn_bias']

        # print('backbone.layer4 ', sum([p.numel() for p in self.layer4.parameters() if p.requires_grad]))
        # print('layer5 ', sum([p.numel()
        #                       for p in self.layer5.parameters() if p.requires_grad]))
        # print('layer5.assp ', sum(
        #     [p.numel() for p in self.layer5.assp.parameters() if p.requires_grad]))

    def train(self, mode=True):
        super(DeepLabV3Plus3, self).train(mode)

        if not self._accum_batch_norm_stats:
            for n, m in self.named_modules():
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
        outputs = super(DeepLabV3Plus3, self).forward(inputs_padded)
        outputs = F.upsample(outputs, size=(h, w), mode='bilinear')

        if crop[0]:
            outputs = outputs[..., crop[0]:]
        if crop[1]:
            outputs = outputs[..., : -crop[1]]
        if crop[2]:
            outputs = outputs[:,:, crop[2]:]
        if crop[3]:
            outputs = outputs[:,:, : -crop[3]]

        return [outputs]
