# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class NChannelResNet(models.ResNet):
    """Wrapper around torchvision.models.ResNet for arbitrary numbers of input channels"""

    def __init__(self, block_type, layers, num_input_channels):
        super().__init__(block_type, layers)
        self.num_enc_channels = [64, 64, 128, 256, 512]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )



class PoseNet(NChannelResNet):
    """Wraps around NChannelResNet to do full end-to-end pose extraction

    Produces poses as pairs of axis-angle and translation vectors. The axis-angle vector can be interpreted
    as a rotation transformation in the following way:
    - interpret the unit vector pointing in the direction as the axis-angle vector as axis of rotation
    - interpret the magnitude of the axis-angle vector as the angle to rotate
    """

    def __init__(self, block_type, layers, num_input_channels, num_predictions=2):
        super().__init__(block_type, layers, num_input_channels)
        self.num_predictions = num_predictions

        self.squeeze = nn.Conv2d(self.num_enc_channels[-1], 256, 1)
        self.pose0 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pose1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pose2 = nn.Conv2d(256, 6 * self.num_predictions, 1)

    def forward(self, x):
        # ResNet feature extractor bits
        x = self.conv1(x[0])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # pose-specific things
        x = self.squeeze(x)
        x = self.relu(x)
        x = self.pose0(x)
        x = self.relu(x)
        x = self.pose1(x)
        x = self.relu(x)
        x = self.pose2(x)

        x = x.mean(3).mean(2)
        x = 0.01 * x.view(-1, self.num_predictions, 1, 6)

        axisangle = x[..., :3]
        translation = x[..., 3:]

        return axisangle, translation

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
