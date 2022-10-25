import torch
import torch.nn as nn
import torch.nn.functional as F
# Model for ANN-ResNet
# It is based on a single-timestep network using ReLU as activation function.

thresh = 0.5  # 0.5 # neuronal threshold
num_classes = 1000
time_window = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# membrane potential update


class mem_update(nn.Module):
    # ReLU
    def __init__(self):
        super(mem_update, self).__init__()

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        for i in range(time_window):
            mem = x[i]
            spike = F.relu(mem)
            output[i] = spike
        return output


class batch_norm_2d(nn.Module):
    """TDBN"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class Snn_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        super(Snn_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        weight = self.weight
        h = (input.size()[3]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        w = (input.size()[4]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device)
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return c1


######################################################################################################################


class BasicBlock_18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            Snn_Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            batch_norm_2d(out_channels),
            mem_update(),
            Snn_Conv2d(out_channels, out_channels * BasicBlock_18.expansion, kernel_size=3, padding=1, bias=False),
            batch_norm_2d(out_channels * BasicBlock_18.expansion),
            )
        # shortcut
        self.shortcut = nn.Sequential(
            )

        if stride != 1 or in_channels != BasicBlock_18.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels, out_channels * BasicBlock_18.expansion, kernel_size=1, stride=stride, bias=False),
                batch_norm_2d(out_channels * BasicBlock_18.expansion),
            )
        self.mem_update = mem_update()

    def forward(self, x):
        return self.mem_update(self.residual_function(x) + self.shortcut(x))


class ResNet_origin_18(nn.Module):
    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()
        k = 1
        self.in_channels = 64 * k
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3, 64*k, kernel_size=7, padding=3, bias=False, stride=2),
            batch_norm_2d(64*k),
            )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.mem_update = mem_update()
        self.conv2_x = self._make_layer(block, 64*k, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 128*k, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256*k, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512*k, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion*k, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        input = torch.zeros(time_window, x.size()[0], 3, x.size()[2], x.size()[3], device=device)
        for i in range(time_window):
            input[i] = x
        output = self.conv1(input)
        output = self.mem_update(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0)/output.size()[0]
        output = self.fc(output)
        return output


def resnet18():
    return ResNet_origin_18(BasicBlock_18, [2, 2, 2, 2])


def resnet34():
    return ResNet_origin_18(BasicBlock_18, [3, 4, 6, 3])
