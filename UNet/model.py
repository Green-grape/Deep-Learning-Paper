import os
import numpy as np

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_batch_relu(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        # Contracting path
        self.c1 = conv_batch_relu(1, 64)
        self.c2 = conv_batch_relu(64, 64)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = conv_batch_relu(64, 128)
        self.c4 = conv_batch_relu(128, 128)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = conv_batch_relu(128, 256)
        self.c6 = conv_batch_relu(256, 256)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c7 = conv_batch_relu(256, 512)
        self.c8 = conv_batch_relu(512, 512)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c9 = conv_batch_relu(512, 1024)
        self.c10 = conv_batch_relu(1024, 1024)

        # Expansive path
        self.u1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c11 = conv_batch_relu(1024, 512)
        self.c12 = conv_batch_relu(512, 512)

        self.u2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c13 = conv_batch_relu(512, 256)
        self.c14 = conv_batch_relu(256, 256)

        self.u3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c15 = conv_batch_relu(256, 128)
        self.c16 = conv_batch_relu(128, 128)

        self.u4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c17 = conv_batch_relu(128, 64)
        self.c18 = conv_batch_relu(64, 64)
        self.fc = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def crop_and_concat(self, input_, crop_input):
        x_shape = input_.shape
        crop_shape = crop_input.shape
        cropped = crop_input[:, :, (crop_shape[2]-x_shape[2])//2:(crop_shape[2]+x_shape[2]) //
                             2, (crop_shape[3]-x_shape[3])//2:(crop_shape[3]+x_shape[3])//2]
        return torch.cat((input_, cropped), dim=1)

    def forward(self, x):
        # print('x shape', x.shape)
        # Contracting path
        c1 = self.c1(x)
        # print('c1 shape', c1.shape)
        c2 = self.c2(c1)
        # print('c2 shape', c2.shape)
        p1 = self.p1(c2)
        # print('p1 shape', p1.shape)

        c3 = self.c3(p1)
        # print('c3 shape', c3.shape)
        c4 = self.c4(c3)
        # print('c4 shape', c4.shape)
        p2 = self.p2(c4)
        # print('p2 shape', p2.shape)

        c5 = self.c5(p2)
        # print('c5 shape', c5.shape)
        c6 = self.c6(c5)
        # print('c6 shape', c6.shape)
        p3 = self.p3(c6)
        # print('p3 shape', p3.shape)

        c7 = self.c7(p3)
        # print('c7 shape', c7.shape)
        c8 = self.c8(c7)
        # print('c8 shape', c8.shape)
        p4 = self.p4(c8)
        # print('p4 shape', p4.shape)

        c9 = self.c9(p4)
        # print('c9 shape', c9.shape)
        c10 = self.c10(c9)
        # print('c10 shape', c10.shape)

        # Expansive path
        u1 = self.u1(c10)
        # print('u1, c8 shape', u1.shape, c8.shape)
        u1 = self.crop_and_concat(u1, c8)
        c11 = self.c11(u1)
        c12 = self.c12(c11)

        u2 = self.u2(c12)
        u2 = self.crop_and_concat(u2, c6)
        c13 = self.c13(u2)
        c14 = self.c14(c13)

        u3 = self.u3(c14)
        u3 = self.crop_and_concat(u3, c4)
        c15 = self.c15(u3)
        c16 = self.c16(c15)

        u4 = self.u4(c16)
        u4 = self.crop_and_concat(u4, c2)
        c17 = self.c17(u4)
        c18 = self.c18(c17)

        out = self.fc(c18)
        return out
