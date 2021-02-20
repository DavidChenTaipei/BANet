from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.nn import functional as F
from torch.autograd import Variable



class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer): #, up_kwargs):
        super(StripPooling, self).__init__()
        #print('in channel',in_channels)
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/2)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, strid, padding, dilation, groups, bias: bool = True, padding_mode)
        
        ##dilated 1 is the 1x1 conv
        self.dilated_up_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 1), 1, (0, 0),dilation=1, bias=False),
                                norm_layer(inter_channels))
        self.dilated_up_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 2),dilation=2, bias=False),
                                norm_layer(inter_channels))
        self.dilated_up_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 5), 1, (0, 8),dilation=4, bias=False),
                                norm_layer(inter_channels))
        
        self.dilated_down_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 1), 1, (0, 0),dilation=1, bias=False),
                                norm_layer(inter_channels))
        self.dilated_down_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (2, 0),dilation=2, bias=False),
                                norm_layer(inter_channels))
        self.dilated_down_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (5, 1), 1, (8, 0),dilation=4, bias=False),
                                norm_layer(inter_channels))
        ## make the 2C channel back to 1C
        self.conv_channel = nn.Sequential(nn.Conv2d(3*inter_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels))
        #self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
        #                        norm_layer(inter_channels))
        #self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
        #                        norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, c, h, w = x.size()
        #print('in h',h)
        #print('in w',w)
        
        #print('x size',x.size())
        
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        
        x2 = self.pool3(x2)
        
        #x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_4_1 = self.dilated_up_1(x2)
        x2_4_3 = self.dilated_up_3(x2)
        x2_4_5 = self.dilated_up_5(x2)
        
        x2_4_3c =torch.cat([x2_4_1,x2_4_3,x2_4_5],dim=1)
        x2_4 = F.interpolate(F.relu_(x2_4_3c), (h, w), **self._up_kwargs)
        x2_4 = self.conv_channel(x2_4)
        '''print('x2_4_1 shape',x2_4_1.shape)
        print('x2_4_3 shape',x2_4_3.shape)
        print('x2_4_5 shape',x2_4_5.shape)
        print('x2_4 shape', x2_4.shape)
        print('x2_4 shape', x2_4.shape)'''
        #x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x2_5_1 = self.dilated_down_1(x2)
        x2_5_3 = self.dilated_down_3(x2)
        x2_5_5 = self.dilated_down_5(x2)
        x2_5_3c = torch.cat([x2_5_1,x2_5_3,x2_5_5],dim=1)
        x2_5 = F.interpolate(F.relu_(x2_5_3c), (h, w), **self._up_kwargs)
        x2_5 = self.conv_channel(x2_5)
        '''print('x2_5_1 shape',x2_5_1.shape)
        print('x2_5_3 shape',x2_5_3.shape)
        print('x2_5_5 shape',x2_5_5.shape)
        print('x2_5 shape', x2_5.shape)'''
        
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(x2_5 + x2_4)
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu((x + out))















class SPHead(nn.Module): ##NOT Going To Use This, it's part of SPNet
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                norm_layer(inter_channels),
                nn.ReLU(True)
        )
        self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                norm_layer(inter_channels // 2),
                nn.ReLU(True),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(inter_channels // 2, out_channels, 1))

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        return x
