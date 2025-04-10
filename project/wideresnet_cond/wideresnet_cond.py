# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .wideresnet import get_norm, conv3x3, Identity


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, norm=None, leak=.2):
        super(wide_basic, self).__init__()
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.temb_dense = nn.Linear(512, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        x, temb = x
        out = self.bn1(x)
        out = self.conv1(self.lrelu(out))
        if temb is not None:
            # add in timestep embedding
            temp_o = self.lrelu(self.temb_dense(temb))
            b, l = temp_o.shape
            out += temp_o.view(b, l, 1, 1)

        out = self.dropout(out)
        out = self.bn2(out)
        out = self.conv2(self.lrelu(out))
        out += self.shortcut(x)

        return out, temb


class Wide_ResNet_Cond(nn.Module):
    def __init__(self, depth, widen_factor, cond_dim, num_classes, input_channels=3, norm=None, leak=.2, dropout_rate=0.0):
        super(Wide_ResNet_Cond, self).__init__()
        self.leak = leak
        self.in_planes = 16
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)
        self.n_classes = num_classes

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        print('| Wide-Resnet %dx%d, time embedding' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k, 128 * k]
        print("nStages", nStages)

        self.layer_one_out = None
        self.conv1 = conv3x3(input_channels, nStages[0])
        # self.layer_one = self.conv1
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=2, leak=leak)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, leak=leak)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, leak=leak)
        self.layer4 = self._wide_layer(wide_basic, nStages[4], n, dropout_rate, stride=2, leak=leak)
        self.bn1 = get_norm(nStages[4], self.norm)
        self.last_dim = nStages[4]
        self.linear = nn.Linear(nStages[4], num_classes)
        self.cond_0 = nn.Linear(cond_dim, 512)
        self.cond_1 = nn.Linear(512, 512)
        self.cond_2 = nn.Linear(512, nStages[4])

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, leak=0.2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, leak=leak, norm=self.norm))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, cond):
        out = self.conv1(x)
        assert x.dtype == torch.float32
        cond = self.cond_0(cond)
        cond = self.cond_1(self.lrelu(cond))
        out, _ = self.layer1([out, cond])
        out, _ = self.layer2([out, cond])
        out, _ = self.layer3([out, cond])
        out, _ = self.layer4([out, cond]) 
        out = self.lrelu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        cond = self.cond_2(self.lrelu(cond))
        out *= cond
        out = self.linear(out)
        return out
    
    def load_pretrained(self, ckpt):
        self.load_state_dict(ckpt, strict=False)




if __name__ == '__main__':
    device = "cuda:0"
    net = Wide_ResNet_Cond(10, 8, 16, num_classes = 1024, input_channels=2, norm='layer', dropout_rate=0.0).to(device)
    x = torch.randn([64, 2, 128, 128]).to(device)
    z = torch.randn([64, 16]).to(device)
    output = net(x, z)
    print(output.shape)
