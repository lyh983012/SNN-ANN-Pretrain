# -*- coding: utf-8 -*-
# python3 example_cifar10_scnn.py
import sys
from importlib import import_module
from util import *
import util
import torch,time,os,random
import torch.nn as nn
import torch.nn.functional as F
import LIAF_module
import LIAF


################################ parameters ####################

batch_size  = 25
num_epochs = 100
learning_rate = 1e-3
timeWindows = 6
names = 'dvs_cifar10'
device = LIAF.device

train_path = r'/home/lyh/dataset/cifar/train.mat' #r := raw string
test_path =  r'/home/lyh/dataset/cifar/test.mat' #TODO:input your oath

train_dataset = cifar10_DVS(train_path,'r',timeWindows) #max= 25?
test_dataset = cifar10_DVS(test_path,'r',timeWindows)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

LIAF.lens = 0.3

modules = import_module('models.LIAFCNN')
config = modules.Config()

config.cfgFc =[512,10]
config.cfgCnn=[(2, 32, 3, 1, 1 ,True),
(32, 64, 3, 1, 1 ,True),
(64, 128, 3, 1, 1 ,True),
(128, 256, 3, 1, 1 ,True)]
config.decay = 0.7
config.dropOut= 0.1
config.timeWindows = timeWindows
config.actFun=torch.selu
config.useBatchNorm = True
config.useLayerNorm = False
config.useThreshFiring = False
config.padding=0
config.dataSize=42
config._data_sparse= True
snn = modules.LIAFCNN(config).to(config.device)
device_ids=[0,1,2,3,4,5,6,7]
snn = nn.DataParallel(snn,device_ids=device_ids)

best_acc = 0
acc_record = list([])

criterion = nn.MSELoss()

#optimizer = torch.optim.SGD(snn.parameters(), lr = learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(snn.parameters(),lr = learning_rate, weight_decay = 1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    train_total = 0
    train_correct = 0 

    #training
    snn.train(mode=True)
    for i, (images, labels) in enumerate(train_loader):
        snn.train(mode=True)
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.to(device)
        outputs = snn(images)  # todo
        loss = criterion(outputs.cpu(), labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted.cpu() == labels).sum()

        if (i+1)%10 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
            running_loss = 0
            print('train acc: %.4f' % (100 * train_correct.float() / train_total))
            train_total = 0
            train_correct = 0
    print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    scheduler.step() # 学习率衰减
    optimizer = lr_scheduler(optimizer,epoch,learning_rate,40)

    #evaluation
    snn.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            outputs = snn(images)  # todo
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        print('Iters:', epoch,'\n\n\n')
        print('Test Acc: %.3f' % (100 * correct.float() / total))
        acc = 100.*correct/total
        acc_record.append(float(acc.data))
        if acc > best_acc:
            print('Saving..')
            best_acc = acc
        print(acc_record)
