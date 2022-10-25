import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from torch.autograd import Variable
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
thresh = 0.5 # neuronal threshold
lens = 0.5   # hyper-parameters of approximate function
decay = 0.5 # decay constants
num_classes = 1000
batch_size  = 20
learning_rate = 3e-2
num_epochs = 75 # max epoch
time_window=8
is_training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = math.sqrt(5)
def assign_optimizer(model, lrs=1e-3):

    rate = 1
    fc1_params = list(map(id, model.fc1.parameters()))
    #fc2_params = list(map(id, model.fc2.parameters()))
    # fc3_params = list(map(id, model.fc3.parameters()))
    base_params = filter(lambda p: id(p) not in fc1_params  , model.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': model.fc1.parameters(), 'lr': lrs * rate},
      #  {'params': model.fc2.parameters(), 'lr': lrs * rate},
          ]
        , lr=lrs,momentum=0.9)

    print('successfully reset lr')
    return optimizer
# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(x):
    mem = torch.zeros_like(x,dtype=torch.half)
    spike = torch.zeros_like(x,dtype=torch.half)
    output = torch.zeros_like(x,dtype=torch.half)
    for i in range(time_window):
        if i>=1 :
            mem[i] = mem[i-1].clone()*decay*(1 - spike[i-1].clone()) + x[i].clone()
        else:
            mem[i] = mem[i].clone()*decay  + x[i].clone()
        spike[i] = act_fun(mem[i].clone()) 
    return spike

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


class batch_norm_2d(nn.Module):
    def __init__(self,num_features,eps=1e-5,momentum=0.1):
        super(batch_norm_2d,self).__init__()
        self.bn = nn.BatchNorm3d(num_features)
    def forward(self,input):
        y = input.transpose(0,2).contiguous().transpose(0,1).contiguous()
        y = self.bn(y,4)
        return y.contiguous().transpose(0,1).contiguous().transpose(0,2)
class batch_norm_2d1(nn.Module):
    def __init__(self,num_features,eps=1e-5,momentum=0.1):
        super(batch_norm_2d1,self).__init__()
        self.bn = nn.BatchNorm3d(num_features)
    def forward(self,input):
        y = input.transpose(0,2).contiguous().transpose(0,1).contiguous()
        y = self.bn(y,8)
        return y.contiguous().transpose(0,1).contiguous().transpose(0,2)

class Snn_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        super(Snn_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        h = (input.size()[3]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        w = (input.size()[4]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        c1 = torch.zeros(time_window,input.size()[1],self.out_channels , h, w,dtype=torch.half,device=input.device)
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return c1

class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.conv1 = Snn_Conv2d(in_ch,out_ch,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = batch_norm_2d(out_ch)
        self.conv2 = Snn_Conv2d(out_ch,out_ch,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = batch_norm_2d1(out_ch)
        self.bn3 = batch_norm_2d1(out_ch)
        self.right = shortcut

    def forward(self,input):
        out = self.conv1(input)
        out = self.bn1(out)
        out = mem_update(out)
        out = self.conv2(out)
        out = self.bn2(out)
        

        residual = self.bn3(input) if self.right is None else self.right(input)
        out += residual
        out = mem_update(out)
        return out
class Snn_ResNet34(nn.Module):

    def __init__(self,num_class=1):
        super(Snn_ResNet34,self).__init__()
        self.pre_conv = Snn_Conv2d(2,64,7,stride=2,padding=3,bias=False)
        self.pre_bn = batch_norm_2d(64)
        self.layer1 = self.make_layer(64,64,2,stride=2)
        self.layer2 = self.make_layer(64,128,2,stride=2)
        self.layer3 = self.make_layer(128,256,2,stride=2)
        self.layer4 = self.make_layer(256,512,2,stride=2)

        self.fc1 = nn.Linear(512,1000)
 

    def make_layer(self,in_ch,out_ch,block_num,stride=1):
        shortcut = nn.Sequential(
            Snn_Conv2d(in_ch,out_ch,1,stride,bias=False),
            batch_norm_2d1(out_ch)
            )
        layers = []
        layers.append(ResidualBlock(in_ch,out_ch,stride,shortcut))
        for i in range(1,block_num):
            layers.append(ResidualBlock(out_ch,out_ch))
        torch.cuda.empty_cache()
        return nn.Sequential(*layers)

    def forward(self,input_):
        torch.cuda.empty_cache()
        input = torch.zeros(time_window,input_.size()[0],2,224,224,dtype=torch.half,device=device)
        for i in range(time_window):
            input[i]=input_[:,i,:,:,:]
        input = self.pre_conv(input)
        input = self.pre_bn(input)
        out = mem_update(input)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)
        

        features = torch.zeros(time_window,input_.size()[0],512,1,1,dtype=torch.half,device=device)
        for i in range(time_window):
             features[i] = F.avg_pool2d(out[i],7)
        features = features.view(time_window,input_.size()[0],-1)
        features = features.sum(dim=0)
        out1 = self.fc1(features /time_window)
        torch.cuda.empty_cache()
        return out1

