# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

class PaddedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        return self.conv(self.pad(x))


class UpsampleSkipConv(nn.Module):
    def __init__(self, in_channels, intermediate_channels, skip_channels, out_channels):
        super().__init__()
        self.padconv0 = PaddedConv(in_channels, intermediate_channels)
        self.padconv1 = PaddedConv(intermediate_channels + skip_channels, out_channels)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")
        self.elu = nn.ELU(inplace=True)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x, skip=None):
        # upsample lower resolution tensor to skip resolution
        x = self.padconv0(x)
        x = self.elu(x)
        x = [self.upsample(x)]

        # concatenate skip features
        if skip is not None:
            x += [skip]
        x = torch.cat(x, 1)

        # apply more convolutions
        x = self.padconv1(x)
        x = self.elu(x)

        return x


class DepthNet(models.ResNet):
    """U-Net style autoencoder based on ResNet"""

    def __init__(self, block_type, layers):
        super().__init__(block_type, layers)
        # encoder layers inherited
        self.num_enc_channels = [64, 64, 128, 256, 512]
        self.num_dec_channels = [16, 32, 64, 128, 256]

        # decoder stuff
        self.decoder = OrderedDict()
        for layer in range(4, -1, -1):
            in_channels = (
                self.num_enc_channels[-1]
                if layer == 4
                else self.num_dec_channels[layer + 1]
            )
            intermediate_channels = self.num_dec_channels[layer]
            skip_channels = self.num_enc_channels[layer - 1] if layer > 0 else 0
            out_channels = self.num_dec_channels[layer]
            self.decoder[f"conv{layer}"] = UpsampleSkipConv(
                in_channels,
                intermediate_channels,
                skip_channels,
                out_channels,
            )

            if layer != 4:
                # for actually predicting disparity
                self.decoder[f"disparity_conv{layer}"] = PaddedConv(
                    self.num_dec_channels[layer], 1
                )

        self.decoder_layers = nn.ModuleList(list(self.decoder.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoder
        feats = []
        x = self.conv1(x)
        x = self.bn1(x)
        feats.append(self.relu(x))
        feats.append(self.layer1(self.maxpool(feats[-1])))
        feats.append(self.layer2(feats[-1]))
        feats.append(self.layer3(feats[-1]))
        feats.append(self.layer4(feats[-1]))

        # decoder
        outputs = {}
        x = feats[-1]
        for layer in range(4, -1, -1):
            skip = feats[layer - 1] if layer > 0 else None
            x = self.decoder[f"conv{layer}"](x, skip)

            if layer != 4:
                outputs[("disp", layer)] = self.sigmoid(
                    self.decoder[f"disparity_conv{layer}"](x)
                )
        return outputs
