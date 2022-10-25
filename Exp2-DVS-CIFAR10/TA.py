import torch
from torch import nn
#from einops import rearrange
import torch.nn.functional as F

timeWindows = 8

class Tlayer(nn.Module):
    '''
    Temporal-wise Attention Layer
    '''
    def __init__(self, timeWindows, reduction=5, dimension=3):
        super(Tlayer, self).__init__()
        if dimension == 3:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
        elif dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)
            
        self.temporal_excitation = nn.Sequential(nn.Linear(timeWindows, int(timeWindows // reduction)),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(int(timeWindows // reduction), timeWindows),
                                                 nn.Sigmoid()
                                                 )

    def forward(self, input):
        b = input.size(0)
        t = input.size(1)

        temp = self.avg_pool(input)
        y_a = temp.view(b, t)
        temp = self.max_pool(input)
        y_m = temp.view(b, t)
        y_a = self.temporal_excitation(y_a).view(temp.size())
        y_m = self.temporal_excitation(y_m).view(temp.size())
        y = torch.sigmoid(y_a+y_m)
        y = torch.mul(input, y)

        return y

class CA_Block(nn.Module):
    def __init__(self, h, w, channel, timeWindows, dimension=3, reduction=5):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w
        self.c = channel

        # self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        # self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.avg_pool_x = nn.AdaptiveAvgPool3d((1, h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool3d((1, 1, w))
        # self.avg_pool_c = nn.AdaptiveAvgPool3d((channel, 1, 1))

        self.conv_1x1 = nn.Conv3d(in_channels=timeWindows, out_channels=timeWindows//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()

        self.F_h = nn.Conv3d(in_channels=timeWindows//reduction, out_channels=timeWindows, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv3d(in_channels=timeWindows//reduction, out_channels=timeWindows, kernel_size=1, stride=1, bias=False)
        # self.F_c = nn.Conv3d(in_channels=timeWindows//reduction, out_channels=timeWindows, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        # self.sigmoid_c = nn.Sigmoid()

    def forward(self, x):
        b = x.shape[0]
        # x = rearrange(x, 'b f c h w -> (b f) c h w')

        x_h = self.avg_pool_x(x).permute(0, 1, 2, 4, 3)
        x_w = self.avg_pool_y(x)
        # x_c = self.avg_pool_c(x).permute(0, 1, 4, 3, 2)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 4)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 4)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 2, 4, 3)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        # s_c = self.sigmoid_c(self.F_c(x_cat_conv_split_c.permute(0, 1, 4, 3, 2)))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        # out = rearrange(out, '(b f) c h w -> b f c h w', b=b)
        return out

class TimeAttention(nn.Module):
    def __init__(self, timeWindows, ratio=4):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(timeWindows, timeWindows // ratio, 1, bias=False), 
            nn.ReLU(),
            nn.Conv3d(timeWindows // ratio, timeWindows, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=5):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False), 
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = rearrange(x, 'b f c h w -> b c f h w')
        x = x.permute(0,2,1,3,4)
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        out = out.permute(0,2,1,3,4)
        out = rearrange(out, 'b c f h w -> b f c h w')
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b = x.size(0)
        h = x.size(3)
        w = x.size(4)
        #x = rearrange(x, 'b f c h w -> b (f c) h w')
        x = x.view(b,-1,h,w)
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = x.unsqueeze(1)
        
        return self.sigmoid(x)


class TCSA(nn.Module):

    def __init__(self, timeWindows, channels, stride=1):
        super(TCSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out

class TCA(nn.Module):

    def __init__(self, timeWindows, channels, stride=1):
        super(TCA, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        out = self.ca(out) * out  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        return out

class CSA(nn.Module):

    def __init__(self, timeWindows, channels, stride=1):
        super(CSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        self.ca = ChannelAttention(channels)
        # self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        out = self.ca(x) * x  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out

class TSA(nn.Module):

    def __init__(self, timeWindows, channels, stride=1):
        super(TSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        # self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out

class TA(nn.Module):

    def __init__(self, timeWindows, channels, stride=1):
        super(TA, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        # self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        return out

class CA(nn.Module):

    def __init__(self, timeWindows, channels, stride=1):
        super(CA, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        self.ca = ChannelAttention(channels)
        # self.ta = TimeAttention(timeWindows)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        out = self.ca(x) * x  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        return out

class SA(nn.Module):

    def __init__(self, timeWindows, channels, stride=1):
        super(SA, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        # self.ca = ChannelAttention(channels)
        # self.ta = TimeAttention(timeWindows)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        # out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        out = self.sa(x) * x  # 广播机制

        out = self.relu(out)
        return out

class TimeAttention_(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(TimeAttention_, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.sharedMLP = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False), nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class TA_(nn.Module):

    def __init__(self, timeWindows, channels, stride=1):
        super(TA_, self).__init__()

        self.relu = nn.ReLU(inplace=True)


        # self.ca = ChannelAttention(channels)
        self.ta = TimeAttention_(timeWindows)
        # self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x
        # out = self.ca(x) * out  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        return out