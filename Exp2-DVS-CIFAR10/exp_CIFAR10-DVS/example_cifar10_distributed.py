# cd SNN_work/genunit_opt/exp_ES_imagenet
# CUDA_VISIBLE_DEVICES=4,5,6,7  python -m torch.distributed.launch --nproc_per_node=4 example_cifar10_large.py
from __future__ import print_function
import argparse, pickle, torch, time, os,sys,math
sys.path.append("..")
import LIAF
import TA
from LIAFnet.LIAFResNet import *
from tensorboardX import SummaryWriter
from datasets.cifar10_dvs_large_10 import cifar10_DVS
from importlib import import_module
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist 
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tensorboardX import SummaryWriter
from importlib import import_module
from LIAFnet.LIAFResNet import *
autocast = LIAF.autocast #统一autocast模式


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

        
# 配置LIAF包
args = {'allow_print':True,
        'use_gause_approx':False,
        'use_rect_approx':True,
        'decay_trainable':False,
        'thresh_trainable':False,
        'use_td_batchnorm':False,
        'if_clamp_the_output':False,
        'save_featuremap':False,
        'seed_value':1
       }
LIAF.config_LIAF(args)


#仅在master进程上输出
writer = None 
master = False 
save_folder = 'spikingResNet18_1110_T=8_LIF'#任务：训练ReLU1下的ResNet，精度大于65%
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',type = int,default=0)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group('nccl',init_method='env://')
local_rank = dist.get_rank()
world_size = dist.get_world_size()
print(dist.get_rank(),' is ready')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################### Step2. load in dataset #####################

modules = import_module('LIAFnet.LIAFResNet_18')
config  = modules.Config()
##########################################################
# 修改部分2
# 激活函数
##########################################################
#config.actFun= torch.nn.ReLU()
#Step2. LIF 训练50epoch

learning_rate = 1e-2

config.actFun= LIAF.LIFactFun.apply
config.block = LIAFResBlock
workpath = os.path.abspath(os.getcwd())
config.cfgCnn = [2, 64, 7, True]
config.dataSize = (128,128)
config.batch_size = 24 #Time = 1
config.num_epochs = 100
batch_size = config.batch_size
num_epochs = config.num_epochs
timeWindows = 8
TA.timeWindows = timeWindows #用了attention必须固定时间长度
config.attention_model = None


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
training_iter = 0
start_epoch = 0
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

names = 'dvs_cifar10'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path = '/data/CIFAR10-MAT-20' #r := raw string
train_dataset = cifar10_DVS(train_path,'train',step=1) 
test_dataset = cifar10_DVS(train_path,'test',step=1)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8,pin_memory=True,drop_last=True,sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=True)

##################### Step3. establish module #####################

snn = LIAFResNet(config)
snn=torch.nn.SyncBatchNorm.convert_sync_batchnorm(snn)
snn.to(device)

##########################################################
# 修改部分3
# 载入模型
##########################################################
print('using uniformed init')
pretrain_path = './45.pkl'
checkpoint = torch.load(pretrain_path, map_location=torch.device('cpu'))
snn.load_state_dict(checkpoint)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(snn.parameters(),
                    lr=learning_rate,
                    weight_decay =1e-4)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-8)

#防止进程冲突
with torch.cuda.device(local_rank):
    snn = DDP(snn,device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    
if local_rank == 0 :
    writer = SummaryWriter(comment='DistributedT=20')
    master = True
    print('no bugs, start recording')
    
    
################step4. training and validation ################
bestacc = 0
def val(optimizer,snn,test_loader,test_dataset,batch_size,epoch):
    print('===> evaluating models...')
    snn.eval()
    correct = 0
    total = 0
    predicted = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if ((batch_idx+1)<=len(test_dataset)//batch_size):
                optimizer.zero_grad()
                outputs = snn(inputs.type(LIAF.dtype))
                labels = targets.squeeze(1).cpu()
                _ , predict = outputs.max(1)
                total += float(labels.size(0))
                correct += float(predict.cpu().eq(labels).sum())
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    print('============>acc:',acc)
    if master:
        writer.add_scalar('acc', acc,epoch)
    return acc


for epoch in range(num_epochs):
    
    #timeWindows = math.floor(epoch/5)+1
    running_loss = 0
    correct = 0.0
    total = 0.0

    snn.train()
    start_time = time.time() 
    print('===> training models...')
    torch.cuda.empty_cache()
    train_loader.sampler.set_epoch(epoch)
    for i, (images, labels) in enumerate(train_loader):
        if ((i+1)<=len(train_dataset)//batch_size):
            outputs = snn(images.to(device)).cpu()
            labels = labels.squeeze(1)
            _ , predict = outputs.max(1)
            loss = criterion(outputs, labels)
            correct += predict.eq(labels).sum()
            total += float(predict.size(0))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1)%10 == 0:
                if master : 
                    if not os.path.isdir(save_folder):
                        os.mkdir(save_folder)
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f \n'
                    %(epoch+start_epoch, num_epochs, i+1, len(train_dataset)//(world_size*batch_size),running_loss ))
                    print('Time elasped: %d \n'  %(time.time()-start_time))
                    writer.add_scalar('running Loss', running_loss, training_iter)
                    train_acc =  correct / total
                    print('Epoch [%d/%d], Step [%d/%d], acc: %.5f \n'
                        %(epoch+start_epoch, num_epochs, i+1, len(train_dataset)//(world_size*batch_size), train_acc)) 
                    writer.add_scalar('running Acc', train_acc*100, training_iter)
                correct = 0.0
                total = 0.0
                running_loss = 0
        training_iter +=1 
    lr_scheduler.step()
    with torch.no_grad():
        if master:
            acc = val(optimizer,snn,test_loader,test_dataset,batch_size,epoch)
            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)
            if acc > bestacc:
                bestacc = acc
                print('===> Saving models...')
                torch.save(snn.module.state_dict(),
                         './'+save_folder+'/'+str(int(bestacc))+'_t='+str(int(timeWindows))+'.pkl')
            writer.add_scalar('bestacc', bestacc, epoch)

