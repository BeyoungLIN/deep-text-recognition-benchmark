
import os
import sys

from torch import nn

class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers, page_orient='horizontal'):
        super(ResNet, self).__init__()
        # self.blur = blur
        self.page_orient = page_orient

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[1],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        if self.page_orient == 'horizontal':
            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        elif self.page_orient == 'vertical':
            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(1, 2), padding=(1, 0))

        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[2],
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)

        if self.page_orient == 'horizontal':
            self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                     kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        elif self.page_orient == 'vertical':
            self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                     kernel_size=2, stride=(1, 2), padding=(1, 0), bias=False)

        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[3],
                                 kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # input - (batch_size, channel, H, W)
    def forward(self, x):
        x = self.conv0_1(x)  # (batch_size, 32, H, W)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)  # (batch_size, 64, H, W)
        x = self.bn0_2(x)
        x = self.relu(x)

        # x = self.maxpool1(x)  # (batch_size, 64, H/2, W/2)
        x = self.layer1(x)  # (batch_size, 128, H/2, W/2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.maxpool2(x)  # (batch_size, 128, H/4, W/4)
        x = self.layer2(x)  # (batch_size, 256, H/4, W/4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # x = self.maxpool3(x)  # (batch_size, 256, H/8, W/4+1) or (batch_size, 256, H/4+1, W/8)
        x = self.layer3(x)  # (batch_size, 512, H/8, W/4+1) or (batch_size, 512, H/4+1, W/8)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)  # (batch_size, 512, H/16, W/4+2) or (batch_size, 512, H/4+2, W/16)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)  # (batch_size, 512, H/16-1, W/4+1) or (batch_size, 512, H/4+1, W/16-1)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x
