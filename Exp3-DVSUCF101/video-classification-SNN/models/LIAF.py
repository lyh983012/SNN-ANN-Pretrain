import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
import random
# pytorch>1.6.0
from torch.cuda.amp import autocast

dtype = torch.float

#######################################################
# activation
# LIFactFun : approximation firing function
# For 2 value-quantified approximation of δ(x)
# LIF激活函数
#######################################################
class LIFactFun(torch.autograd.Function):
    lens = 0.5  # LIF激活函数的梯度近似参数，越小则梯度激活区域越窄
    bias = -0.2  # 多阈值激活函数的值域平移参数
    sigma = 1  # 高斯梯度近似时的sigma
    use_rect_approx = False  # 选择方梯度近似方法
    use_gause_approx = True  # 选择高斯梯度近似方法

    def __init__(self):
        super(LIFactFun, self).__init__()

    # 阈值激活，带有输入阈值，阈值可训练
    @staticmethod
    def forward(ctx, input, thresh=0.5):
        fire = input.gt(thresh).float()
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return fire

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = 0
        thresh = ctx.thresh
        if LIFactFun.use_rect_approx:
            temp = abs(input - thresh) < LIFactFun.lens
            grad = grad_input * temp.float() / (2 * LIFactFun.lens)
        elif LIFactFun.use_gause_approx:
            temp = 0.3989422804014327 / LIFactFun.sigma * torch.exp(
                -0.5 / (LIFactFun.sigma ** 2) * (input - thresh + LIFactFun.bias) ** 2)
            grad = grad_input * temp.float()
        return grad, None


#######################################################
# cells
# LIF神经元
# 所有参数都可训练的LIF神经元
# 阈值按ARLIF的训练方法
# decay使用sigmoid归一化
# FC处理2维输入，Conv处理4维输入
#######################################################
class BaseNeuron(nn.Module):
    fire_rate_upperbound = 0.8
    fire_rate_lowebound = 0.2
    thresh_trainable = False
    decay_trainable = True
    use_td_batchnorm = False
    save_featuremap = False

    def __init__(self):
        super().__init__()
        self.device = None
        self.norm = 0.3
        self.fire_rate = None
        self.actFun = LIFactFun.apply
        self.thresh = 0.5
        self.decay = torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=BaseNeuron.decay_trainable)

    def forward(self, input):
        raise NotImplementedError("Input neurons must implement `base_neuron_forward`")

    def norm_decay(self, decay):
        # decay调整方法——BP自适应+限制
        return torch.sigmoid(decay)

    def cal_fire_rate_and_thresh_update(self, newV, spike):
        # ARLIF 论文实现的阈值调整方法
        inputsize = newV.size()
        count = 1.0
        for dim in inputsize:
            count = count * dim
        self.fire_rate = spike.sum() / count
        if BaseNeuron.thresh_trainable:
            if self.fire_rate > BaseNeuron.fire_rate_upperbound:
                self.thresh += 0.1
            if self.fire_rate < BaseNeuron.fire_rate_lowebound:
                self.thresh -= 0.1

    # 输入突触连接为conv层 输入维度为[B,C,T,H,W] 或者 [B,T,N]
    def mem_update(self, x, init_mem=None, spikeAct=LIFactFun.apply):

        dim = len(x.size())
        output = torch.zeros_like(x).to(self.device)
        mem_old = 0

        if dim == 5:
            spike = torch.zeros_like(x[:, :, 0, :, :]).to(self.device)
            time_window = x.size(2)
            mem = x[:, :, 0, :, :]
        elif dim == 3:
            spike = torch.zeros_like(x[:, 0, :]).to(self.device)
            time_window = x.size(1)
            mem = x[:, 0, :]
        if init_mem is not None:
            mem = init_mem
        for i in range(time_window):
            if i >= 1 and dim == 5:
                mem = mem_old * self.norm_decay(self.decay) * (1 - spike.detach()) + x[:, :, i, :, :]
            elif i >= 1 and dim == 3:
                mem = mem_old * self.norm_decay(self.decay) * (1 - spike.detach()) + x[:, i, :]
            mem_old = mem.clone()
            spike = spikeAct(mem_old, self.thresh)
            self.cal_fire_rate_and_thresh_update(mem_old, spike)
            if dim == 3:
                if self.actFun == LIFactFun.apply:
                    output[:, i, :] = self.actFun(mem_old, self.thresh)
                else:
                    output[:, i, :] = self.actFun(mem_old)
            elif dim == 5:
                if self.actFun == LIFactFun.apply:
                    output[:, :, i, :, :] = self.actFun(mem_old, self.thresh)
                else:
                    output[:, :, i, :, :] = self.actFun(mem_old)
        return output


#######################################################
# 【复合的LIAF神经元，突触连接：linear】
# standard LIAF cell based on LIFcell
# 简介: 最简单的LIAF cell，由LIF模型的演变而来
# 记号：v为膜电位，f为脉冲，x为输入，w为权重，t是时间常数
#         v_t' = v_{t-1} + w * x_n
#         f = spikefun(v_t')
#         x_{n+1} = analogfun(v_t')
#         v_t = v_t' * (1-f) * t
# 用法：torch_network.add_module(name,LIAF.LIAFCell())
# update: 2020-02-29
# author: Linyh

# update: 2021-09-02
# self.only_linear for last fc layer in CNN-encoder
#######################################################
class LIAFCell(BaseNeuron):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 actFun=torch.relu,
                 dropOut=0,
                 useBatchNorm=False,
                 only_linear=False   # for the last fc layer in CNN-encoder (which should be set True)
                 ):
        '''
        @param input_size: (Num) number of input
        @param hidden_size: (Num) number of output
        @param actfun: handle of activation function to hidden layer(analog fire)
        @param decay: time coefficient in the model
        @param dropout: 0~1
        @param useBatchNorm: use batch-norm (treat every input[t] as a sample)
        '''
        super().__init__()
        self.device = None
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.actFun = actFun
        self.spikeActFun = LIFactFun.apply
        self.useBatchNorm = useBatchNorm
        self.batchSize = None
        self.timeWindows = None
        # block 1：add synaptic inputs: Wx+b=y
        self.kernel = nn.Linear(inputSize, hiddenSize)
        # block 2： add a BN layer(optional)
        if self.useBatchNorm:
            self.NormLayer = nn.BatchNorm1d(hiddenSize)
        # block 3： use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:
            self.UseDropOut = True

        self.featuremap = None  # 存储中间输出的Feature Map 用于可视化
        self.fire_rate = None  # 存储激活率
        self.thresh = 0.5
        self.decay = torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=BaseNeuron.decay_trainable)

        self.only_linear = only_linear

    def forward(self,
                data,
                init_v=None):
        """
        @param input: a tensor of of shape (Batch, time, insize)
        @param init_v: a tensor with size of (Batch, time, outsize) denoting the mambrane v.
        """
        # step 0: init
        self.device = data.device
        self.batchSize = data.size(0)  # adaptive for mul-gpu training
        self.timeWindows = data.size(1)
        synaptic_input = torch.zeros(self.batchSize, self.timeWindows, self.hiddenSize, device=self.device)
        # Step 1: accumulate
        for time in range(self.timeWindows):
            synaptic_input[:, time, :] = self.kernel(data[:, time, :].view(self.batchSize, -1))
            # if self.useBatchNorm:
            #     synaptic_input[:, time, :] = self.NormLayer(synaptic_input[:, time, :])
        if BaseNeuron.save_featuremap:
            self.featuremap = synaptic_input  # 暂存特征图
        if self.useBatchNorm:
            for time in range(self.timeWindows):
                synaptic_input[:, time, :] = self.NormLayer(synaptic_input[:, time, :])
        # Step 3: update membrane
        output = synaptic_input
        if not self.only_linear:
            output = self.mem_update(output)
        # step 4: DropOut
        if self.UseDropOut:
            output = self.DPLayer(output)

        return output


#######################################################
# 【复合的LIAF神经元，突触连接：linear】
# standard LIAF-RNN cell based on LIAFcell
# 简介: 仿照RNN对LIAF模型进行轻度修改
# 记号：v为膜电位，f为脉冲，x为输入，w1w2为权重，t是时间常数
#         v_t' = w1 *v_{t-1} + w2 * x_n
#         f = spikefun(v_t')
#         x_{n+1} = analogfun(v_t')
#         v_t = v_t' * (1-f) * t
# update: 2020-03-21
# author: Linyh
#######################################################
class LIAFRCell(BaseNeuron):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 actFun=torch.selu,
                 dropOut=0,
                 useBatchNorm=False
                 ):
        '''
        @param input_size: (Num) number of input
        @param hidden_size: (Num) number of output
        @param actfun: handle of activation function
        @param decay: time coefficient in the model
        @param dropout: 0~1
        @param useBatchNorm: if use
        '''
        super().__init__()
        self.device = None
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize

        self.actFun = actFun
        self.timeWindows = None
        self.spikeActFun = LIFactFun.apply
        self.useBatchNorm = useBatchNorm
        self.UseDropOut = True
        self.batchSize = None

        # block 1. add synaptic inputs: Wx+b=y
        self.kernel = nn.Linear(inputSize, hiddenSize)
        self.kernel_v = nn.Linear(hiddenSize, hiddenSize)

        # block 2. add a BN layer
        if self.useBatchNorm:
            self.NormLayer = nn.BatchNorm1d(hiddenSize)

        # block 3. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

    def forward(self,
                input,
                init_v=None):
        """
        @param input: a tensor of of shape (Batch, time, insize)
        @param init_v: a tensor with size of (Batch, time, outsize) denoting the mambrane v.
        """
        # step 0: init
        sum_of_elements = 1
        for i in input.size():
            sum_of_elements *= i
        self.fire_rate = input.sum() / sum_of_elements

        self.device = input.device
        self.timeWindows = input.size(1)

        self.batchSize = input.size()[0]  # adaptive for mul-gpu training
        output_init = torch.zeros(self.batchSize, self.timeWindows, self.hiddenSize, device=self.device)
        output_fired = torch.zeros(self.batchSize, self.timeWindows, self.hiddenSize, device=self.device)
        # Step 1: accumulate
        for time in range(self.timeWindows):
            event_frame_t = input[:, time, :].float().to(self.device)
            event_frame_t = event_frame_t.view(self.batchSize, -1)
            output_init[:, time, :] = self.kernel(event_frame_t)
        # Step 2: Normalization
        normed_v = output_init
        if self.useBatchNorm:
            normed_v = self.NormLayer(output_init)

        if init_v is None:
            v = torch.zeros(self.batchSize, self.hiddenSize, device=event_frame_t.device)
        else:
            v = init_v
        for time in range(self.timeWindows):
            v = normed_v[:, time, :] + self.kernel_v(v)
            # Step 2: Fire and leaky
            fire = self.spikeActFun(v)
            output = self.actFun(v)
            output_fired[:, time, :] = output
            v = self.decay * (1 - fire) * v
        # step 4: DropOut
        if self.UseDropOut:
            output_fired = self.DPLayer(output_fired)

        return output_fired


#######################################################
# 复合的LIAF神经元，突触连接：2dconv
# standard LIAF cell based on LIFcell
# 简介: 替换线性为卷积的 cell，由LIF模型的演变而来
# 记号：v为膜电位，f为脉冲，x为输入，w为权重，t是时间常数
#         v_t' = v_{t-1} + w conv x_n
#         f = spikefun(v_t')
#         x_{n+1} = analogfun(v_t')
#         v_t = v_t' * (1-f) * t
# update: 2020-02-29
# author: Linyh
#######################################################
class LIAFConvCell(BaseNeuron):

    def __init__(self,
                 inChannels,
                 outChannels,
                 kernelSize,
                 stride,
                 padding=(0, 0),
                 dilation=1,
                 groups=1,
                 actFun=torch.relu,
                 dropOut=0,
                 useBatchNorm=True,
                 inputSize=(224, 224),
                 usePool=False,
                 p_method='max',
                 p_kernelSize=2,
                 p_stride=2,
                 p_padding=0
                 ):
        '''
        @param inChannels: (Num) number of input Channels
        @param outChannels: (Num) number of output Channels
        @param kernelSize: size of convolutional kernel
        @param p_kernelSize: size of pooling kernel
        @param stride，padding，dilation，groups -> for convolutional input connections
        @param outChannels: (Num) number of output
        @param actfun: handle of activation function to hidden layer(analog fire)
        @param decay: time coefficient in the model
        @param Qbit: >=2， # of thrsholds of activation
        @param dropout: 0~1
        @param useBatchNorm: if use batch-norm
        @param p_method: 'max' or 'avg'
        @param act: designed for resnet
        '''
        super().__init__()
        self.device = None
        self.kernelSize = kernelSize
        self.actFun = actFun
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.spikeActFun = LIFactFun.apply
        self.useBatchNorm = useBatchNorm
        self.usePool = usePool
        self.inputSize = inputSize
        self.batchSize = None
        self.outChannel = outChannels
        self.layer_index = 0
        self.p_method = p_method
        self.p_stride = p_stride
        self.p_kernelSize = p_kernelSize
        self.p_padding = p_padding
        self.timeWindows = None
        # block 1. conv layer:
        self.kernel = nn.Conv2d(inChannels,
                                outChannels,
                                self.kernelSize,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                groups=self.groups,
                                bias=True,
                                padding_mode='zeros')
        # block 2. add a pooling layer

        # block 3. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:
            self.UseDropOut = True  # enable drop_out in cell

        # Automatically calulating the size of feature maps
        self.CSize = list(self.inputSize)
        self.CSize[0] = math.floor((self.inputSize[0] + 2 * self.padding[0]
                                    - kernelSize[0]) / self.stride[0] + 1)
        self.CSize[1] = math.floor((self.inputSize[1] + 2 * self.padding[1]
                                    - kernelSize[1]) / self.stride[1] + 1)
        self.outputSize = list(self.CSize)

        if self.useBatchNorm:
            self.NormLayer = nn.BatchNorm3d(outChannels)

        if self.usePool:
            self.outputSize[0] = math.floor((self.outputSize[0] + 2 * self.p_padding
                                             - self.p_kernelSize) / self.p_stride + 1)
            self.outputSize[1] = math.floor((self.outputSize[1] + 2 * self.p_padding
                                             - self.p_kernelSize) / self.p_stride + 1)

        self.featuremap = None  # 存储中间输出的Feature Map 用于可视化
        self.fire_rate = None  # 存储激活率
        self.thresh = 0.5
        self.decay = torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=BaseNeuron.decay_trainable)
        self.featuremap_pooled = None

    def forward(self,
                input,
                init_v=None):
        """
        @param input: a tensor of of shape (B, C, T, H, W)
        @param init_v: an tensor denoting initial mambrane v with shape of  (B, C, T, H, W). set None use 0
        @return: new state of cell and output
        note : only batch-first mode can be supported
        """
        # step 0: init
        self.timeWindows = input.size(2)
        self.batchSize = input.size(0)
        self.device = input.device

        synaptic_input = torch.zeros(self.batchSize, self.outChannel, self.timeWindows,
                                     self.CSize[0], self.CSize[1], device=self.device)

        # Step 1: accumulate
        for time in range(self.timeWindows):
            synaptic_input[:, :, time, :, :] = self.kernel(input[:, :, time, :, :].to(self.device))
        if BaseNeuron.save_featuremap:
            self.featuremap = synaptic_input

        # Step 2: Normalization
        if self.useBatchNorm:
            synaptic_input = self.NormLayer(synaptic_input)

        # Step 3: LIF
        output = self.mem_update(synaptic_input)

        # Step 4: Pooling
        if self.usePool:
            output_pool = torch.zeros(self.batchSize, self.outChannel, self.timeWindows,
                                      self.outputSize[0], self.outputSize[1], device=self.device)
            for time in range(self.timeWindows):
                if self.p_method == 'max':
                    output_pool[:, :, time, :, :] = F.max_pool2d(output[:, :, time, :, :],
                                                                 kernel_size=self.p_kernelSize,
                                                                 stride=self.p_stride, padding=self.p_padding)
                else:
                    output_pool[:, :, time, :, :] = F.avg_pool2d(output[:, :, time, :, :],
                                                                 kernel_size=self.p_kernelSize,
                                                                 stride=self.p_stride, padding=self.p_padding)
        else:
            output_pool = output
        if BaseNeuron.save_featuremap:
            self.featuremap_pooled = synaptic_input

        # step 5: DropOut
        if self.UseDropOut:
            output_pool = self.DPLayer(output_pool)

        return output_pool

