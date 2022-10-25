# import torch
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
from LIAF import LIAFCell, LIAFConvCell


class LIAFCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.actFun = config['actFun']
        self.timeWindows = config['time_windows']
        self.useBatchNorm = config['useBatchNorm']
        self.cfg_cnn = config['cfg_cnn']
        self.cfg_fc = config['cfg_fc']
        self.nCnnLayers = len(config['cfg_cnn'])
        self.network = nn.Sequential()
        # self.useThreshFiring = config.useThreshFiring
        # self._data_sparse = config._data_sparse
        # self.if_static_input = config.if_static_input
        # self.onlyLast = config.onlyLast
        self.batchSize = None

        self.dataSize = config['input_size']

        for dice in range(self.nCnnLayers):
            inChannels, outChannels, kernelSize, stride, padding, usePool, p_kernelSize, p_stride = self.cfg_cnn[dice]
            CNNlayer = LIAFConvCell(inChannels=inChannels,
                                    outChannels=outChannels,
                                    kernelSize=kernelSize,
                                    stride=stride,
                                    p_kernelSize=p_kernelSize,
                                    padding=padding,
                                    actFun=self.actFun,
                                    usePool=usePool,
                                    inputSize=self.dataSize,
                                    useBatchNorm=self.useBatchNorm)
            self.dataSize = CNNlayer.outputSize  # renew the fearture map size
            self.network.add_module('cnn' + str(dice), CNNlayer)

        self.cfg_fc_ = [outChannels * self.dataSize[0] * self.dataSize[1]]
        self.cfg_fc_.extend(self.cfg_fc)
        self.nFcLayer = len(self.cfg_fc_) - 1  # special
        for dice2 in range(self.nFcLayer):  # DO NOT REUSE LIAFMLP!!! BIGBUG
            only_linear = False
            if dice2 == self.nFcLayer - 1:
                only_linear = True
            self.network.add_module('fc' + str(dice2 + dice + 1),
                                    LIAFCell(self.cfg_fc_[dice2],
                                             self.cfg_fc_[dice2 + 1],
                                             actFun=self.actFun,
                                             useBatchNorm=False,
                                             only_linear=only_linear))

    def forward(self, x):
        # output x: (batch_size, time_window, n_nodes)
        self.batchSize = x.size(0)
        # self.timeWindows = x.size(2)

        for layer in self.network:
            if isinstance(layer, LIAFCell):
                x = x.view(self.batchSize, self.timeWindows, -1)
            x = layer(x)

        return x
