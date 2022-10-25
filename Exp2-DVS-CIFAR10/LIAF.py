#data:2021-04-29
#author:linyh
#email: 532109881@qq.com
#note: 增加了各层的阈值的定制能力，便于实现optimal conversion方法
#  修改部分：LIAFactfun的thresh改成计算时传入
#  修改部分：mem_update根据输入维度统一
import torch
import torch.nn as nn 
import torch.nn.functional as F
import os
import math
import numpy as np
import random

seed_value = 1   # 设定随机数种子
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
torch.manual_seed(seed_value)           # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）
torch.backends.cudnn.deterministic = True
dtype = torch.float
allow_print = False

############################################
# update 07-26
# 加入注意力机制
############################################
import TA

############################################
# autocast 需要pytorch>3.6.0
############################################
import sys
__USE_AUTO_CAST__ = False 
if sys.version_info.minor>=6:
    __USE_AUTO_CAST__ = True
    print('use torch.cuda.amp.autocast for fusion prcision training')

def autocast():
    class null_cast:
        def __enter__(self):
            pass
        def __exit__(self,exc_type,exc_val,exc_tb):
            pass
    if __USE_AUTO_CAST__:
        from torch.cuda.amp import autocast
        return autocast()
    else:
        return null_cast()

############################################
# update 08-30
# 加入全局配置接口
############################################    
def config_LIAF(args):
    if len(args)==0:
        return 0
    # 配置激活函数
    if 'use_rect_approx' in args:
        LIFactFun.use_rect_approx = args['use_rect_approx']
        LIFactFun.use_gause_approx = not LIFactFun.use_rect_approx
    if 'use_gause_approx' in args:
        LIFactFun.use_gause_approx = args['use_gause_approx']
        LIFactFun.use_rect_approx = not LIFactFun.use_gause_approx
        
    # 配置基本神经元
    if 'thresh_trainable' in args:
        BaseNeuron.thresh_trainable = args['thresh_trainable']
    if 'decay_trainable' in args:
        BaseNeuron.decay_trainable = args['decay_trainable']
    if 'use_td_batchnorm' in args:
        BaseNeuron.use_td_batchnorm = args['use_td_batchnorm']
    if 'if_clamp_the_output' in args:
        BaseNeuron.if_clamp_the_output = args['if_clamp_the_output']
    if 'save_featuremap' in args:
        BaseNeuron.save_featuremap = args['save_featuremap']
        
    if 'allow_print' in args:
        allow_print = args['allow_print']
    if 'dtype' in args:
        dtype = args['dtype']
        
    if 'seed_value' in args:
        seed_value = args['seed_value']
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
        torch.manual_seed(seed_value)           # 为CPU设置随机种子
        torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
        torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）
        torch.backends.cudnn.deterministic = True

#######################################################
# activation
# LIFactFun : approximation firing function
# For 2 value-quantified approximation of δ(x)
# LIF激活函数
#######################################################    

class LIFactFun(torch.autograd.Function):
    lens = 0.5      # LIF激活函数的梯度近似参数，越小则梯度激活区域越窄
    bias = -0.2     # 多阈值激活函数的值域平移参数                            
    sigma = 1       # 高斯梯度近似时的sigma
    use_rect_approx = False # 选择方梯度近似方法【Switch Flag】
    use_gause_approx = True # 选择高斯梯度近似方法【Switch Flag】
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
            grad = grad_input*temp.float()/(2*LIFactFun.lens)
        elif LIFactFun.use_gause_approx:
            temp = 0.3989422804014327 / LIFactFun.sigma*torch.exp(-0.5/(LIFactFun.sigma**2)*(input-thresh+LIFactFun.bias)**2) 
            grad = grad_input * temp.float()
        return grad, None

#######################################################
#init method
#######################################################
def paramInit(model,method='kaiming'):
    scale = 0.05
    if isinstance(model, nn.BatchNorm3d) or isinstance(model, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    else:
        for name, w in model.named_parameters():
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.orthogonal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0.5)
            else:
                pass
            
#######################################################
# cells 
# LIF神经元
# 所有参数都可训练的LIF神经元
# 阈值按ARLIF的训练方法
# decay使用sigmoid归一化
# FC处理2维输入，Conv处理4维输入
# update: 2021-08-25
# author: Linyh
#######################################################
class BaseNeuron(nn.Module):
    
    # For advanced training【Switch Flag】
    fire_rate_upperbound = 0.8
    fire_rate_lowebound  = 0.2
    thresh_trainable = False
    decay_trainable = True
    use_td_batchnorm = False 
    
    # For Debug【Switch Flag】
    if_clamp_the_output = True
    save_featuremap = False
    
    def __init__(self):
        super().__init__()
        
        self.norm = 0.3
        self.fire_rate = None
        self.actFun = LIFactFun.apply
        self.thresh =  0.5
        self.decay =torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=BaseNeuron.decay_trainable)
        self.act = True

    def forward(self,input):
        raise NotImplementedError("Input neurons must implement `base_neuron_forward`")

    def norm_decay(self,decay):
        # decay调整方法——BP自适应+限制
        return torch.sigmoid(decay)
    
    def cal_fire_rate_and_thresh_update(self,newV,spike):
        # ARLIF 论文实现的阈值调整方法
        inputsize = newV.size()
        count = 1.0
        for dim in inputsize:
            count = count*dim
        self.fire_rate = spike.sum()/count
        if BaseNeuron.thresh_trainable: 
            if self.fire_rate> BaseNeuron.fire_rate_upperbound:
                self.thresh += 0.1
            if self.fire_rate< BaseNeuron.fire_rate_lowebound:
                self.thresh -= 0.1
          
    # 输入突触连接为conv层 输入维度为[B,C,T,W,H] 或者 [B,T,N]
    def mem_update(self,x,init_mem=None,spikeAct = LIFactFun.apply):
        dim = len(x.size())
        if dim == 5:
            output = self.__5Dmem_update(x,init_mem,spikeAct)
        if dim == 3:
            output = self.__3Dmem_update(x,init_mem,spikeAct)      
        if BaseNeuron.if_clamp_the_output:#[beta] 实验ReLU1
            output = output.clamp_(0.0, 1.0)
        return output
          
    # 输入突触连接为conv层 输入维度为[B,C,T,W,H] 或者 [B,T,N]
    def __5Dmem_update(self,x,init_mem=None,spikeAct = LIFactFun.apply):
        mem_old= 0
        time_window = x.size(2)
        mem = x[:,:,0,:,:]
        outputs = torch.zeros_like(x)
        if init_mem is not None:
            mem = init_v 
        for t in range(time_window):
            if t>=1:
                mem = mem_old * self.norm_decay(self.decay)*(1 - spike.detach()) + x[:,:,t,:,:]
            mem_old = mem.clone()
            spike = spikeAct(mem_old,self.thresh) 
            self.cal_fire_rate_and_thresh_update(mem_old,spike)
            if self.act:
                if self.actFun == LIFactFun.apply:
                    output = self.actFun(mem_old,self.thresh)
                else:
                    output = self.actFun(mem_old)
            else:
                output = mem_old
            outputs[:,:,t,:,:] = output

        return outputs
    
    def __3Dmem_update(self,x,init_mem=None,spikeAct = LIFactFun.apply):
        mem_old= 0
        time_window = x.size(1)
        mem = x[:,0,:]
        outputs = torch.zeros_like(x)
        if init_mem is not None:
            mem = init_v 
        for t in range(time_window):
            if t>=1:
                mem = mem_old * self.norm_decay(self.decay)*(1 - spike.detach()) + x[:,t,:]
            mem_old = mem.clone()
            spike = spikeAct(mem_old,self.thresh) 
            self.cal_fire_rate_and_thresh_update(mem_old,spike)
            if self.act:
                if self.actFun == LIFactFun.apply:
                    output = self.actFun(mem_old,self.thresh)
                else:
                    output = self.actFun(mem_old)
            else:
                output = mem_old
            outputs[:,t,:] = output
        return outputs
    
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
# update: 2021-08-20
# author: Linyh
#######################################################
class LIAFCell(BaseNeuron):
    
    def __init__(self,
        inputSize,
        hiddenSize,
        actFun = torch.selu,
        dropOut= 0,
        useBatchNorm = False,
        init_method='kaiming'
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
        self.inputSize = inputSize              
        self.hiddenSize = hiddenSize       
        self.actFun = actFun                
        self.spikeActFun = LIFactFun.apply  
        self.useBatchNorm = useBatchNorm    
        self.batchSize = None
        self.timeWindows = None
        # block 1：add synaptic inputs: Wx+b=y
        self.kernel=nn.Linear(inputSize, hiddenSize)    
        paramInit(self.kernel,init_method)
        # block 2： add a BN layer(optional)
        if self.useBatchNorm:
            self.NormLayer = nn.BatchNorm1d(hiddenSize)
        # block 3： use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)             
        if 0 < dropOut < 1: 
            self.UseDropOut = True
        
        self.featuremap = None # 存储中间输出的Feature Map 用于可视化
        self.fire_rate = None  # 存储激活率
        self.thresh =  0.5
        self.decay = torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=BaseNeuron.decay_trainable)
       
        
    def forward(self,
        data,
        init_v=None):
        """
        @param input: a tensor of of shape (Batch, time, insize)
        @param init_v: a tensor with size of (Batch, time, outsize) denoting the mambrane v.
        """
        #step 0: init
        with autocast():
            self.device = self.kernel.weight.device
            if data.device != self.device:
                data = data.to(self.device)
            self.batchSize = data.size(0)#adaptive for mul-gpu training
            self.timeWindows = data.size(1)
            synaptic_input = torch.zeros(self.batchSize,self.timeWindows,self.hiddenSize,device=self.device,dtype=dtype)
            # Step 1: accumulate 
            for time in range(self.timeWindows):
                synaptic_input[:,time,:] = self.kernel(data[:,time,:].view(self.batchSize, -1))
                if self.useBatchNorm:
                    synaptic_input[:,time,:] = self.NormLayer(synaptic_input[:,time,:])
            if BaseNeuron.save_featuremap:
                self.featuremap = synaptic_input # 暂存特征图
            if self.useBatchNorm:
                for time in range(self.timeWindows):
                    synaptic_input[:,time,:] = self.NormLayer(synaptic_input[:,time,:]) 
            # Step 3: update membrane
            output = self.mem_update(synaptic_input)
            # step 4: DropOut
            if self.UseDropOut:
                output = self.DPLayer(output)
        return output
    

#######################################################    
# 【复合的LIAF神经元，突触连接：linear】
#standard LIAF-RNN cell based on LIAFcell
#简介: 仿照RNN对LIAF模型进行轻度修改
#记号：v为膜电位，f为脉冲，x为输入，w1w2为权重，t是时间常数
#         v_t' = w1 *v_{t-1} + w2 * x_n
#         f = spikefun(v_t')
#         x_{n+1} = analogfun(v_t')
#         v_t = v_t' * (1-f) * t
#update: 2021-08-23
#author: Linyh
#######################################################
class LIAFRCell(BaseNeuron):
    def __init__(self,
        inputSize,
        hiddenSize,
        actFun = torch.selu,
        dropOut= 0,
        useBatchNorm = False,
        init_method='kaiming'
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
        self.inputSize = inputSize              
        self.hiddenSize = hiddenSize       
        self.actFun = actFun                
        self.spikeActFun = LIFactFun.apply  
        self.useBatchNorm = useBatchNorm    
        self.batchSize = None
        self.timeWindows = None
        # block 1：add synaptic inputs: Wx+b=y
        self.kernel=nn.Linear(inputSize, hiddenSize)    
        paramInit(self.kernel,init_method)
        self.kernel_v = nn.Linear(hiddenSize, hiddenSize)
        paramInit(self.kernel_v,init_method)
        # block 2： add a BN layer(optional)
        if self.useBatchNorm:
            self.NormLayer = nn.BatchNorm1d(hiddenSize)
        # block 3： use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)             
        if 0 < dropOut < 1: 
            self.UseDropOut = True
        
        self.featuremap = None # 存储中间输出的Feature Map 用于可视化
        self.fire_rate = None  # 存储激活率
        self.thresh =  0.5
        self.decay =torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=BaseNeuron.decay_trainable)
        
    def forward(self,
        input,
        init_v=None):
        """
        @param input: a tensor of of shape (Batch, time, insize)
        @param init_v: a tensor with size of (Batch, time, outsize) denoting the mambrane v.
        """
        #step 0: init
        with autocast():
            self.device = self.kernel.weight.device
            if data.device != self.device:
                data = data.to(self.device)
            self.batchSize = data.size(0)#adaptive for mul-gpu training
            self.timeWindows = data.size(1)
            synaptic_input = torch.zeros(self.batchSize,self.timeWindows,self.hiddenSize,device=self.device,dtype=dtype)
            # Step 1: accumulate 
            for time in range(self.timeWindows):
                synaptic_input[:,time,:] = self.kernel(data[:,time,:].view(self.batchSize, -1))
                if self.useBatchNorm:
                    synaptic_input[:,time,:] = self.NormLayer(synaptic_input[:,time,:])
            if BaseNeuron.save_featuremap:
                self.featuremap = synaptic_input # 暂存特征图
            if self.useBatchNorm:
                for time in range(self.timeWindows):
                    synaptic_input[:,time,:] = self.NormLayer(synaptic_input[:,time,:]) 
            # Step 3: update membrane
            if v is None:#initialization of V
                if init_v is None:
                    v = torch.zeros(self.batchSize, self.hiddenSize, device=self.device)
                else:
                    v = init_v   
            for time in range(self.timeWindows):
                v = normed_v[:,time,:] + self.kernel_v(v)
                fire = self.spikeActFun(v)
                output = self.actFun(v)
                output_fired[:,time,:] = output
                v = self.decay * (1 - fire.detach()) * v
            if self.UseDropOut:
                output_fired = self.DPLayer(output_fired)
        return output_fired

    
#######################################################    
# 复合的LIAF神经元，突触连接：2dconv
# standard LIAF cell based on LIFcell
#简介: 替换线性为卷积的 cell，由LIF模型的演变而来
#记号：v为膜电位，f为脉冲，x为输入，w为权重，t是时间常数
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
                 padding =0,
                 dilation=1,
                 groups=1,
                 dropOut =0,
                 inputSize=(224,224),
                 init_method='kaiming',
                 usePool= True,
                 p_method = 'avg',
                 p_kernelSize = 2,
                 p_stride = 2,
                 p_padding = 0,
                 act = True,
                 actFun=LIFactFun.apply,
                 attention_model = None,
                 useBatchNorm = False
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
        @param dropout: 0~1
        @param useBatchNorm: if use batch-norm
        @param p_method: 'max' or 'avg'
        
        @param act: 是否激活
        @param useBatchNorm: 是否批归一化
        @param usePool: 是否池化
        @param __with_lif_neuron: 是否加入膜电位更新（默认True，不在接口中修改）
        @param save_featuremap: 是否存储特征图矩阵（默认False，不在接口中修改）
        
        '''
        super().__init__()
        self.kernelSize = kernelSize
        self.actFun = actFun
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.spikeActFun = LIFactFun.apply
        self.useBatchNorm= useBatchNorm
        self.usePool= usePool
        self.inputSize = inputSize
        self.outChannel = outChannels
        self.layer_index = 0
        self.p_method  = p_method
        self.p_stride = p_stride
        self.p_kernelSize = p_kernelSize
        self.p_padding = p_padding
        self.act = act
        
        self.batchSize = None
        self.timeWindows = None
        
        # block 1. conv layer:
        self.kernel = nn.Conv2d(inChannels,outChannels,self.kernelSize,
                                stride=self.stride,
                                padding= self.padding,
                                dilation=self.dilation,
                                groups=self.groups,
                                bias=True,
                                padding_mode='zeros')
        paramInit(self.kernel,init_method)
        # block 2. add a attetion layer
        
        if attention_model is not None:
            self.att_module = attention_model(timeWindows=TA.timeWindows, channels = outChannels)
        else:
            self.att_module = attention_model
        
        # block 3. use dropout
        self.UseDropOut=False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1: 
            self.UseDropOut = True# enable drop_out in cell
        
        # Automatically calulating the size of feature maps
        self.CSize = list(self.inputSize)
        self.CSize[0] = math.floor((self.inputSize[0] + 2 * self.padding 
            - kernelSize[0]) / stride + 1)
        self.CSize[1] = math.floor((self.inputSize[1] + 2 * self.padding 
            - kernelSize[1]) / stride + 1)
        self.outputSize = list(self.CSize)

        if self.useBatchNorm:
            self.NormLayer = nn.BatchNorm3d(outChannels)
        
        
        if self.usePool:
            self.outputSize[0] = math.floor((self.outputSize[0]+ 2 * self.p_padding 
            - self.p_kernelSize) / self.p_stride + 1)
            self.outputSize[1] = math.floor((self.outputSize[1]+ 2 * self.p_padding 
            - self.p_kernelSize) / self.p_stride + 1)
            
 
        self.thresh =  0.5
        self.decay =torch.nn.Parameter(torch.ones(1) * 0.5, requires_grad=BaseNeuron.decay_trainable)

        self.featuremap = None # 存储中间输出的Feature Map 用于可视化
        self.fire_rate = None  # 存储激活率
        self.featuremap_pooled = None
        self.__with_lif_neuron = True
        
    def forward(self,
                input_data,
                init_v=None):
        """
        @param input: a tensor of of shape (B, C，T, H，W)
        @param init_v: an tensor denoting initial mambrane v with shape of  (B, C，T, H，W). set None use 0
        @return: new state of cell and output
        note : only batch-first mode can be supported 
        """
        #step 0: init
        with autocast():
            self.timeWindows = input_data.size(2)
            self.batchSize = input_data.size(0)
            self.device = self.kernel.weight.device      
            if input_data.device != self.device:
                input_data = input_data.to(self.device) 

            synaptic_inputs = torch.zeros(self.batchSize,self.outChannel, self.timeWindows, 
                        self.CSize[0], self.CSize[1],device=self.device,dtype=dtype)

            # Step 1: conv and accumulate 
            for time in range(self.timeWindows):
                synaptic_inputs[:,:,time,:,:] = self.kernel(input_data[:,:,time,:,:].to(self.device))
   
            # step 1.5: attention
            if self.att_module is not None:
                synaptic_inputs = synaptic_inputs.permute(0,2,1,3,4)
                synaptic_inputs = self.att_module(synaptic_inputs)#(B, T, C, H，W)
                synaptic_inputs = synaptic_inputs.permute(0,2,1,3,4)

            # Step 2: Normalization 
            if self.useBatchNorm:
                synaptic_inputs = self.NormLayer(synaptic_inputs)

            # Step 3: POOLING 
            if self.usePool:
                pooled_inputs = torch.zeros(self.batchSize,self.outChannel, self.timeWindows, 
                                    self.outputSize[0], self.outputSize[1],device=self.device,dtype=dtype)  
                for time in range(self.timeWindows):
                    if self.p_method == 'max':
                        pooled_inputs[:,:,time,:,:] = F.max_pool2d(synaptic_inputs[:,:,time,:,:], kernel_size=self.p_kernelSize,
                        stride = self.p_stride,padding=self.p_padding)
                    else:
                        pooled_inputs[:,:,time,:,:] = F.avg_pool2d(synaptic_inputs[:,:,time,:,:], kernel_size=self.p_kernelSize,
                        stride = self.p_stride,padding=self.p_padding)
            else:
                pooled_inputs = synaptic_inputs
                        
            # Step 4: LIF
            if self.__with_lif_neuron:
                output = self.mem_update(pooled_inputs)

            if BaseNeuron.save_featuremap:    
                self.featuremap_pooled = output

        return output

#######################################################    
# ResBlock for ResNet18/34
# 简介: 基于LIAF-CNN的残差块
# standard LIAF cell based on LIFcell
# update: 2020-08-10
# author: Linyh
#######################################################

class LIAFResBlock(BaseNeuron):
    expansion = 1
    def __init__(self,
                 inChannels,
                 outChannels,
                 kernelSize=(3,3),
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 actFun=LIFactFun.apply,
                 useBatchNorm = False,
                 attention_model = None,
                 inputSize=(224,224),
                 name = 'liafres'
                 ):

        super().__init__()
        self.padding=padding
        self.actFun=actFun
        self.useBatchNorm = useBatchNorm
        self.kernelSize=kernelSize
        self.timeWindows= None
        self.downSample=False
        self.shortcut = None

        if inChannels!=outChannels:
            #判断残差类型——>输入输出是否具有相同维度
            stride = 2
            self.downSample = True
            #print(name +' dimension changed')  
            self.shortcut = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=2)
        else:
            stride = 1

        self.cv1 = LIAFConvCell(inChannels=inChannels,
                                    outChannels=outChannels,
                                    kernelSize=kernelSize,
                                    stride = stride,
                                    padding =1,
                                    dilation= 1,
                                    groups= 1,
                                    inputSize = inputSize,
                                    dropOut =0,
                                    attention_model=attention_model,
                                    useBatchNorm = True,
                                    act = False,
                                    usePool= False)
        self.cv2 = LIAFConvCell(inChannels=outChannels,
                                    outChannels=outChannels,
                                    kernelSize=kernelSize,
                                    stride = 1,
                                    padding = 1,
                                    dilation= 1,
                                    groups=1,
                                    inputSize = self.cv1.outputSize,
                                    dropOut =0,
                                    attention_model=attention_model,
                                    useBatchNorm = True,
                                    act = False,
                                    usePool= False)
        self.cv2.__with_lif_neuron = False
        
        if self.useBatchNorm:
            if BaseNeuron.use_td_batchnorm:
                self.shortcut_norm = thBN.BatchNorm3d(outChannels)
                self.shortcut_norm.k=8
            else:
                self.shortcut_norm = nn.BatchNorm3d(outChannels)
                
        self.outputSize = self.cv2.outputSize
        self.featuremap1 = None
        self.featuremap2 = None
        
        if allow_print:
            print('the output feature map is'+str(self.outputSize))
            
    def forward(self,input_data):
        '''
        设计网络的规则：
        1.对于输出feature map大小相同的层，有相同数量的filters，即channel数相同；
        2. 当feature map大小减半时（池化），filters(channel)数量翻倍。
            对于残差网络，维度匹配的shortcut连接为实线，反之为虚线。维度不匹配时，同等映射有两种可选方案：  
            直接通过zero padding 来增加维度（channel）。
            乘以W矩阵投影到新的空间。实现是用1x1卷积实现的，直接改变1x1卷积的filters数目。这种会增加参数。
        '''
        with autocast():
            self.timeWindows = input_data.size(2)
            self.batchSize = input_data.size(0)
            
            
            cv1_output = self.cv1(input_data)
            cv2_output = self.cv2(cv1_output)

            if self.downSample:
                shortcut_output = torch.zeros(cv2_output.size(),device=cv2_output.device)
                for time in range(self.timeWindows):
                    shortcut_output[:,:,time,:,:] = self.shortcut(input_data[:,:,time,:,:])
                shortcut_output = self.shortcut_norm(shortcut_output)
            else:
                shortcut_output = input_data

            output = self.mem_update(cv2_output+shortcut_output)

        return output

    
#######################################################    
# ResBlock for ResNet18/34
# 简介: 基于LIAF-CNN的残差块
# standard LIAF cell based on LIFcell
# update: 2021-09-02
# author: Linyh & hyf
#######################################################

class LIAFResBlock_LIF(BaseNeuron):
    expansion = 1
    def __init__(self,
                 inChannels,
                 outChannels,
                 kernelSize=(3,3),
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 actFun=LIFactFun.apply,
                 useBatchNorm = False,
                 attention_model = None,
                 inputSize=(224,224),
                 name = 'liafres'
                 ):

        super().__init__()
        self.padding=padding
        self.actFun=actFun
        self.useBatchNorm = useBatchNorm
        self.kernelSize=kernelSize
        self.timeWindows= None
        self.downSample=False
        self.shortcut = None

        if inChannels!=outChannels:
            #判断残差类型——>输入输出是否具有相同维度
            stride = 2
            self.downSample = True
            #print(name +' dimension changed')  
            self.shortcut = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=2)
        else:
            stride = 1

        self.cv1 = LIAFConvCell(inChannels=inChannels,
                                    outChannels=outChannels,
                                    kernelSize=kernelSize,
                                    stride = stride,
                                    padding =1,
                                    dilation= 1,
                                    groups= 1,
                                    inputSize = inputSize,
                                    dropOut =0,
                                    attention_model=attention_model,
                                    useBatchNorm = True,
                                    act = False,
                                    usePool= False)
        self.cv2 = LIAFConvCell(inChannels=outChannels,
                                    outChannels=outChannels,
                                    kernelSize=kernelSize,
                                    stride = 1,
                                    padding = 1,
                                    dilation= 1,
                                    groups=1,
                                    inputSize = self.cv1.outputSize,
                                    dropOut =0,
                                    attention_model=attention_model,
                                    useBatchNorm = True,
                                    act = False,
                                    usePool= False)
        self.cv1.__with_lif_neuron = False
        self.cv2.__with_lif_neuron = False
        
        if self.useBatchNorm:
            if BaseNeuron.use_td_batchnorm:
                self.shortcut_norm = thBN.BatchNorm3d(outChannels)
                self.shortcut_norm.k = 8
            else:
                self.shortcut_norm = nn.BatchNorm3d(outChannels)
                
        self.outputSize = self.cv2.outputSize
        self.featuremap1 = None
        self.featuremap2 = None
        
        if allow_print:
            print('the output feature map is'+str(self.outputSize))
            
    def forward(self,input_data):
        '''
        设计网络的规则：
        1.对于输出feature map大小相同的层，有相同数量的filters，即channel数相同；
        2. 当feature map大小减半时（池化），filters(channel)数量翻倍。
            对于残差网络，维度匹配的shortcut连接为实线，反之为虚线。维度不匹配时，同等映射有两种可选方案：  
            直接通过zero padding 来增加维度（channel）。
            乘以W矩阵投影到新的空间。实现是用1x1卷积实现的，直接改变1x1卷积的filters数目。这种会增加参数。
        '''
        with autocast():

            self.timeWindows = input_data.size(2)
            self.batchSize = input_data.size(0)
            
            cv1_input = self.cv1.mem_update(input_data)
            cv1_output = self.cv1(cv1_input)

            if BaseNeuron.save_featuremap:
                self.featuremap1 = cv1_output

            cv2_input = self.cv2.mem_update(cv1_output)
            cv2_output = self.cv2(cv2_input)

            if self.downSample:
                shortcut_output = torch.zeros_like(cv2_output,device=cv2_output.device)
                for time in range(self.timeWindows):
                    shortcut_output[:,:,time,:,:] = self.shortcut(input_data[:,:,time,:,:])
                shortcut_output = self.shortcut_norm(shortcut_output)
            else:
                shortcut_output = input_data
        
            if BaseNeuron.save_featuremap:
                self.featuremap2 = output

            output = cv2_output+shortcut_output

        return output


#######################################################    
# ResNeck for ResNet50+
# 简介: 基于LIAF-CNN的残差块
# standard LIAF cell based on LIFcell
# update: 2020-11-22
# author: Linyh
#######################################################
class LIAFResNeck(BaseNeuron):
    expansion = 4
    def __init__(self,
                 cahnnel_now,
                 inChannels,
                 kernelSize=(3,3),
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 actFun=LIFactFun.apply,
                 attention_model = None,
                 useBatchNorm = False,
                 useLayerNorm = False,
                 inputSize=(224,224),
                 name = 'liafres'
                 ):

        super().__init__()
        self.padding=padding

        self.actFun=actFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.kernelSize=kernelSize
        self.timeWindows= None
        self.downSample=False
        self.shortcut = None

        if (inChannels* LIAFResNeck.expansion)!=cahnnel_now:
            #判断残差类型——>输入输出是否具有相同维度
            stride = 2
            self.downSample = True
            #print(name +' dimension changed')  
            self.shortcut = nn.Conv2d(cahnnel_now, inChannels* LIAFResNeck.expansion, kernel_size=1, stride=2)
        else:
            stride = 1

        self.cv1 = LIAFConvCell(inChannels=inChannels,
                                    outChannels=outChannels,
                                    kernelSize= (1,1),
                                    stride = stride,
                                    padding =0,
                                    dilation= 1,
                                    groups= 1,
                                    inputSize = inputSize,
                                    dropOut =0,
                                    attention_model=attention_model,
                                    useBatchNorm = True,
                                    act = False,
                                    usePool= False)
        self.cv2 = LIAFConvCell(inChannels=inChannels,
                                    outChannels=outChannels,
                                    kernelSize=kernelSize,
                                    stride = 1,
                                    padding =1,
                                    dilation= 1,
                                    groups= 1,
                                    inputSize = self.cv1.outputSize,
                                    dropOut =0,
                                    attention_model=attention_model,
                                    useBatchNorm = True,
                                    act = False,
                                    usePool= False)
        self.cv3 = LIAFConvCell(inChannels=inChannels,
                                    outChannels=inChannels* LIAFResNeck.expansion,
                                    kernelSize= (1,1),
                                    stride = 1,
                                    padding =0,
                                    dilation= 1,
                                    groups= 1,
                                    inputSize = self.cv2.outputSize,
                                    dropOut =0,
                                    attention_model=attention_model,
                                    useBatchNorm = True,
                                    act = False,
                                    usePool= False)
        self.cv3.__with_lif_neuron = False
        self.outputSize = self.cv3.outputSize

        if self.useBatchNorm:
            self.shortcut_norm = nn.BatchNorm3d(inChannels* LIAFResNeck.expansion)
        if self.useLayerNorm:
            self.shortcut_norm = nn.BatchNorm3d(inChannels* LIAFResNeck.expansion)
        self.featuremap1 = None
        self.featuremap2 = None
        self.featuremap3 = None
        if allow_print:
            print('the output feature map is'+str(self.outputSize))
            
    def forward(self,input):
        '''
        设计网络的规则：
        1.对于输出feature map大小相同的层，有相同数量的filters，即channel数相同；
        2. 当feature map大小减半时（池化），filters(channel)数量翻倍。
            对于残差网络，维度匹配的shortcut连接为实线，反之为虚线。维度不匹配时，同等映射有两种可选方案：  
            直接通过zero padding 来增加维度（channel）。
            乘以W矩阵投影到新的空间。实现是用1x1卷积实现的，直接改变1x1卷积的filters数目。这种会增加参数。
        '''
        with autocast():
            self.timeWindows = input.size(2)
            self.batchSize = input.size(0)

            shortcut_output = input
            cv1_output = self.cv1(input)
            cv2_output = self.cv2(cv1_output)
            cv3_output = self.cv3(cv2_output)
           
            if self.downSample:
                shortcut_output = torch.zeros(cv3_output.size(),device=cv3_output.device)
                for time in range(self.timeWindows):
                    shortcut_output[:,:,time,:,:] = self.shortcut(input[:,:,time,:,:])
                shortcut_output = self.shortcut_norm(shortcut_output)
            output = self.actFun(cv3_output+shortcut_output)
            output = self.mem_update(output)

            if BaseNeuron.save_featuremap:
                self.featuremap1 = cv1_output
                self.featuremap2 = cv2_output
                self.featuremap3 = output
        return output

    
#######################################################    
# 复合的LIAF神经元，以门控形式组合
#LSTMCell
#update: 2020-03-21
#author: Linyh
#简介: 仿照RNN对LIAF模型进行轻度修改，每个门维护一个膜电位
#方案1: 全仿LSTM，当decay=0，spikefire为sigmoid时完全是LSTM
#记号：类似
#         v_t' = v_{t-1} + w2 * x_n
#         f = spikefun(v_t')
#         v_t = v_t' * (1-f) * t
#
#         fi = \sigma(vi) 
#         ff = \sigma(vf) 
#         fg = \tanh(vg)
#         fo = \sigma(vo) 
#         c' = ff * fc + fi * fg 
#         x_{n+1} = o * \tanh(c') 
#######################################################

class LIAFLSTMCell(nn.Module):

    def __init__(self,
        inputSize,
        hiddenSize,
        actFun = torch.selu,
        spikeActFun = LIFactFun.apply,
        decay = 0.3,
        dropOut= 0,
        useBatchNorm= False,
        useLayerNorm= False,
        timeWindows = 5,
        Qbit = 0,
        init_method='kaiming',
        sgFun = torch.relu
        ):
        '''
        @param input_size: (Num) number of input
        @param hidden_size: (Num) number of output
        @param actfun: handle of activation function 
        @param actfun: handle of recurrent spike firing function
        @param decay: time coefficient in the model 
        @param dropout: 0~1 unused
        @param useBatchNorm: unused
        '''

        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.actFun = actFun
        self.sgFun = sgFun #default:sigmoid
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.timeWindows = timeWindows
        self.UseDropOut = True
        self.batchSize = None
        # block 1. add synaptic inputs: Wx+b=y
        self.kernel_i=nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_i,init_method)
        self.kernel_f = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_f,init_method)
        self.kernel_g = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_g,init_method)
        self.kernel_o = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_o,init_method)

        # block 2. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(self.timeWindows)
        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm([self.timeWindows,hiddenSize])
        # block 3. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

        # Choose QLIAF mode
        if Qbit>0 :
            MultiLIFactFun.Nbit = Qbit
            print("warning: assert ActFun = MultiLIFactFun.apply")
            self.actFun = MultiLIFactFun.apply
            print('# of threshold = ', MultiLIFactFun.Nbit)

        self.c =None

    def forward(self,
        input,
        init_v=None):
        """
        @param input: a tensor of of shape (Batch, N)
        @param state: a pair of a tensor including previous output and cell's potential with size (Batch,3, N).
        @return: new state of cell and output for hidden layer
        Dense Layer: linear kernel
        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        x' = o * \tanh(c') \\
        \end{array}
        """
        #step 0: init
        with autocast():
            self.batchSize = input.size()[0]#adaptive for mul-gpu training
            self.device = self.kernel_i.weight.device
            if self.timeWindows != input.size()[1]:
                print('wrong num of time intervals')
            if input.device != self.device:
                input = input.to(self.device)

            if init_v is None:
                vi = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
                vf = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
                vg = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
                vo= torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
                c = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
            else:
                vi = init_v.clone()
                vf = init_v.clone()
                vg = init_v.clone()
                vo= init_v.clone()
                c = init_v.clone()

            output = torch.zeros(self.batchSize,self.timeWindows,self.hiddenSize,device=self.device,dtype=dtype)

            for time in range(self.timeWindows):
            # Step 1: accumulate and reset,spike used as forgetting gate
                vi = self.kernel_i(input[:,time,:].float()) + vi
                vf = self.kernel_f(input[:,time,:].float()) + vf 
                vg = self.kernel_g(input[:,time,:].float()) + vg
                vo= self.kernel_o(input[:,time,:].float()) + vo

                fi = self.spikeActFun(vi)
                ff = self.spikeActFun(vf)
                fg = self.spikeActFun(vg)
                fo = self.spikeActFun(vo)

                # Step 2: figo
                i = self.sgFun(vi)
                f = self.sgFun(vf)
                o = self.sgFun(vo)
                g = self.actFun(vg)

                # step 3: Learn 
                c = c * f  + i * g

                # step 4: renew
                output[:,time,:] = self.actFun(c) * o

                #step 5: leaky
                vi = self.decay * (1 - fi ) * vi
                vf = self.decay * (1 - ff ) * vf
                vg = self.decay * (1 - fg ) * vg
                vo= self.decay * (1 - fo ) * vo

            # step 6: Norms
            if self.useBatchNorm:
                output = self.BNLayerx(output)
            if  self.useLayerNorm:
                output = self.Lnormx(output)

        return output

    
#######################################################    
# 复合的LIAF神经元，以GRU形式组合
# LIAFGRU

    #########
    #author：Lin-Gao
    #in 2020-07
    #########
    
#记号：类似
#         v_t' = v_{t-1} + w2 * x_n
#         f = spikefun(v_t')
#         v_t = v_t' * (1-f) * t
#
#         fi = \sigma(vi) 
#         ff = \sigma(vf) 
#         fg = \tanh(vg)
#         fo = \sigma(vo) 
#         c' = ff * fc + fi * fg 
#         x_{n+1} = o * \tanh(c') 
#######################################################

class LIAFGRUCell(nn.Module):

    def __init__(self, inputSize, hiddenSize, spikeActFun, actFun=torch.selu, dropOut=0,
                 useBatchNorm=False, useLayerNorm=False, init_method='kaiming', gFun=torch.tanh, decay=0.3):
        """
        :param inputSize:(Num) number of input
        :param hiddenSize:(Num) number of output
        :param actFun:handle of activation function
        :param spikeAcFun:handle of recurrent spike firing function
        :param dropOut: 0~1 unused
        :param useBatchNorm:
        :param useLayerNorm:
        :param init_method:
        :param gFun:
        """
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.decay = decay
        self.actFun = actFun
        self.gFun = gFun  # default:tanh
        self.spikeActFun = spikeActFun
        self.useBatchNorm = useBatchNorm
        self.useLayerNorm = useLayerNorm
        self.UseDropOut = True
        self.batchSize = None
        # block 1. add synaptic inputs:Wx+b
        self.kernel_r = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_r, init_method)  # 作用
        self.kernel_z = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_z, init_method)
        self.kernel_h = nn.Linear(inputSize, hiddenSize)
        paramInit(self.kernel_h, init_method)
        # block 2. add synaptic inputs:Hx+b
        self.kernel_r_h = nn.Linear(hiddenSize, hiddenSize)
        paramInit(self.kernel_r_h, init_method)
        self.kernel_z_h = nn.Linear(hiddenSize, hiddenSize)
        paramInit(self.kernel_z_h, init_method)
        self.kernel_h_h = nn.Linear(hiddenSize, hiddenSize, bias=False)
        paramInit(self.kernel_h_h, init_method)
        # block 3. add a Norm layer
        if self.useBatchNorm:
            self.BNLayerx = nn.BatchNorm1d(hiddenSize)
            self.BNLayerc = nn.BatchNorm1d(hiddenSize)
        if self.useLayerNorm:
            self.Lnormx = nn.LayerNorm(hiddenSize)
            self.Lnormc = nn.LayerNorm(hiddenSize)
        # block 4. use dropout
        self.UseDropOut = False
        self.DPLayer = nn.Dropout(dropOut)
        if 0 < dropOut < 1:  # enable drop_out in cell
            self.UseDropOut = True

    def forward(self, input, init_v=None):
        """
        :param input:
        :param init_v:
        :return:
        """
        with autocast():
            self.batchSize = input.size()[0]
            input = input.reshape([self.batchSize, -1])
            if input.device != self.kernel_r.weight.device:
                input = input.to(self.kernel_r.weight.device)
            if self.h is None:
                if init_v is None:
                    self.h = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
                    self.u = torch.zeros(self.batchSize, self.hiddenSize, device=input.device,dtype=dtype)
                else:
                    self.h = init_v.clone()
                    self.u = init_v.clone()
            # Step 1: accumulate and reset,spike used as forgetting gate
            r = self.kernel_r(input.float()) + self.kernel_r_h(self.h)
            z = self.kernel_z(input.float()) + self.kernel_z_h(self.h)
            r = self.actFun(r)
            z = self.actFun(z)
            h = self.kernel_h(input.float()) + self.kernel_h_h(self.h) * r
            h = self.gFun(h)
            # Step 2: renew
            h_ = self.h.clone()
            self.h = self.decay * self.u * (1 - self.spikeActFun(self.u))
            self.u = (1 - z) * h_ + z * h
            x = self.spikeActFun(self.u)

            # step 3: Norms
            if self.useBatchNorm:
                self.h = self.BNLayerc(self.h)
                self.u = self.BNLayerc(self.u)
                x = self.BNLayerx(x)
            if self.useLayerNorm:
                self.h = self.Lnormc(self.h)
                self.u = self.Lnormc(self.u)
                x = self.Lnormx(x)
        return x



    
