
#accept: LSTM和RNN的LIAF化，支持双向
#test: 随机初始化原始膜电位
#data:2020-08-11
#author:linyh
#email: 532109881@qq.com
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import LIAF

#########################################################
'''general_configs'''
#模型demo的初始化在config里统一完成，内含大量默认参数，请认真检查
#参数的修改方式可以在各个example中找到

class Config(object):

    def __init__(self, path=None, dataset=None, embedding=None):
        '''cfg for learning'''
        self.learning_rate = 3e-2                                       # 学习率，最重要的参数，部分demo不是在这里设置
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练，仅LSTM实现
        self.num_epochs =  30                                           # epoch数
        self.batch_size = 16                                           # mini-batch大小，部分demo不是在这里设置
        self.Qbit=0                                                     # 是否使用多阈值函数（>2支持，Qbit的值实际上是阈值个数）
        '''cfg for net'''
        self.num_classes = 1000
        self.cfgCnn = [2,64,7]
        self.cfgRes = [2,2,2,2]
        self.cfgFc = [self.num_classes]
        self.timeWindows = 8
        self.actFun= nn.ReLU()
        self.useBatchNorm = True
        self.useThreshFiring = True
        self._data_sparse= False
        self.padding= 0
        self.inputSize= [224,224]
        
class VGG_ConvSNN(nn.Module):
    def __init__(self,in_channels=3, 
                     out_channels=64, 
                     padding=1, 
                     kernel_size=3, 
                     stride=1, 
                     inputSize = None, 
                     usePool = False,
                     attention_model = None):
        super().__init__()
        self.conv = LIAF.LIAFConvCell(inChannels=in_channels,
                                    outChannels=out_channels,
                                    kernelSize=[kernel_size,kernel_size],
                                    stride= stride,
                                    padding = padding,
                                    actFun=nn.ReLU(),
                                    usePool= False,
                                    useBatchNorm= False,
                                    inputSize= inputSize,
                                    attention_model = attention_model,
                                    p_kernelSize = 2,
                                    p_method = 'max',
                                    p_padding = 0,
                                    p_stride = 2
                                )
        self.outputSize = self.conv.outputSize
    def forward(self,data):
        return self.conv(data)
    
class VGG_LinearSNN(nn.Module):
    def __init__(self,in_features=512*7*7, out_features=4096):
        super().__init__()
        self.fc = LIAF.LIAFCell(in_features,out_features,
                                actFun=nn.ReLU(),
                                dropOut= 0.5 ,
                                useBatchNorm=False)
    def forward(self,data):
        return self.fc(data)

class VGG_SNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        inputSize = config.inputSize
        self.num_classes = config.num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        net.append(VGG_ConvSNN(in_channels=2, out_channels=64, padding=1, 
                               kernel_size=3, stride=1, inputSize=inputSize))
        inputSize = net[-1].outputSize
        net.append(VGG_ConvSNN(in_channels=64, out_channels=64, padding=1, 
                               kernel_size=3, stride=1, inputSize=inputSize,usePool = True))
        inputSize = net[-1].outputSize

        # block 2
        net.append(VGG_ConvSNN(in_channels=64, out_channels=128, kernel_size=3, 
                               stride=1, padding=1, inputSize=inputSize))
        inputSize = net[-1].outputSize
        net.append(VGG_ConvSNN(in_channels=128, out_channels=128, kernel_size=3, 
                               stride=1, padding=1, inputSize=inputSize,usePool = True))
        inputSize = net[-1].outputSize

        # block 3
        net.append(VGG_ConvSNN(in_channels=128, out_channels=256, kernel_size=3, 
                               stride=1, padding=1, inputSize=inputSize))
        inputSize = net[-1].outputSize
        #net.append(VGG_ConvSNN(in_channels=256, out_channels=256, kernel_size=3, 
        #                       stride=1, padding=1, inputSize=inputSize))
        #inputSize = net[-1].outputSize
        net.append(VGG_ConvSNN(in_channels=256, out_channels=256, kernel_size=3, 
                               stride=1, padding=1, inputSize=inputSize,usePool = True))
        inputSize = net[-1].outputSize
        
        # block 4
        net.append(VGG_ConvSNN(in_channels=256, out_channels=512, kernel_size=3, 
                               stride=1, padding=1, inputSize=inputSize))
        inputSize = net[-1].outputSize
        #net.append(VGG_ConvSNN(in_channels=512, out_channels=512, kernel_size=3, 
        #                       stride=1, padding=1, inputSize=inputSize))
        #inputSize = net[-1].outputSize
        net.append(VGG_ConvSNN(in_channels=512, out_channels=512, kernel_size=3, 
                               stride=1, padding=1, inputSize=inputSize,usePool = True))
        inputSize = net[-1].outputSize
        
        # block 5
        net.append(VGG_ConvSNN(in_channels=512, out_channels=512, kernel_size=3, 
                               stride=1, padding=1, inputSize=inputSize))
        inputSize = net[-1].outputSize
        #net.append(VGG_ConvSNN(in_channels=512, out_channels=512, kernel_size=3, 
        #                       stride=1, padding=1, inputSize=inputSize))
        #inputSize = net[-1].outputSize
        net.append(VGG_ConvSNN(in_channels=512, out_channels=512, kernel_size=3, 
                               stride=1, padding=1, inputSize=inputSize,usePool = True))
        inputSize = net[-1].outputSize
        
        # add net into class property
        self.extract_feature = nn.Sequential(*net)

        # define an empty container for Linear operations
        classifier = []
        classifier.append(VGG_LinearSNN(in_features=512*7*7, out_features=4096))
        classifier.append(VGG_LinearSNN(in_features=4096, out_features=4096))
        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)
        
        self.last_fc = nn.Linear(in_features=4096, out_features=self.num_classes)


    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.permute(0,2,1,3,4)
        feature = feature.view(feature.size(0),feature.size(1), -1)
        classify_result = self.classifier(feature)  
        classify_result = classify_result.mean(dim = 1)
        classify_result = self.last_fc(classify_result)                  
                            
        return classify_result
