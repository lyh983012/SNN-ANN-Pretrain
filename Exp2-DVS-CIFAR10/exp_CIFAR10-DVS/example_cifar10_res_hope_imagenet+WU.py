# -*- coding: utf-8 -*-
# python3 example_cifar10_res_hope_imagenet+WU.py
import sys
sys.path.append("..")
from importlib import import_module
import torch,time,os,random
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
import LIAF
from datasets.cifar10_dvs_large_10 import cifar10_DVS
from tensorboardX import SummaryWriter
from importlib import import_module
from LIAFnet.LIAFResNet import *
autocast = LIAF.autocast #统一autocast模式

################################ parameters ####################
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
for test in range(0,10):
    task = 'new_HOPE1_ImageNet_' + str(test)
    writer = SummaryWriter(comment=task)
    learning_rate = 1e-3
    batch_size  = 24
    num_epochs = 60
    timeWindows = 8
    TA.timeWindows = 8 #用了attention必须固定时间长度
    
    names = 'dvs_cifar10'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = '/data/CIFAR10-MAT-20' #r := raw string
    train_dataset = cifar10_DVS(train_path,'train',1) #max= 25?
    test_dataset = cifar10_DVS(train_path,'test',1)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    TA.timeWindows = 8 #用了attention必须固定时间长度
    modules = import_module('LIAFnet.LIAFResNet_18')
    config  = modules.Config()
    workpath = os.path.abspath(os.getcwd())
    accumulation = config.accumulation
    timeWindows = config.timeWindows
    config.cfgCnn = [3, 64, 7, True]
    config.dataSize = (128,128)
    config.actFun= LIAF.LIFactFun.apply
    epoch = 0
    bestacc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    training_iter = 0
    snn = LIAFResNet(config)
    checkpoint = torch.load('./63__t=8-Copy1.pkl', map_location=torch.device('cpu'))
    snn.load_state_dict(checkpoint)
    for p in snn.parameters():
        p.requires_grad=False #TODO
    conv1 = LIAF.LIAFConvCell(inChannels=2,
                             outChannels=snn.cfgCnn[1],
                             kernelSize=[7,7],
                             stride= 2,
                             padding = 3,
                             actFun=snn.actFun,
                             usePool= True,
                             useBatchNorm= snn.useBatchNorm,
                             inputSize= [128,128],
                             p_kernelSize = 3,
                             p_method = 'max',
                             p_padding = 0,
                             p_stride = 2)
    conv1.kernel.weight[:,0,:,:].item = torch.mean(snn.conv1.kernel.weight,dim=1)
    conv1.kernel.weight[:,1,:,:].item = torch.mean(snn.conv1.kernel.weight,dim=1)
    conv1.kernel.bias.item = snn.conv1.kernel.bias.clone()
    conv1.NormLayer.weight.item = snn.conv1.NormLayer.weight.clone()
    conv1.NormLayer.bias.item = snn.conv1.NormLayer.bias.clone
    snn.conv1 = conv1.to(snn.conv1.kernel.bias.device)
    snn.fc = nn.Linear(snn.cfgFc_[0],10)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in snn.parameters())))
    snn.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(snn.parameters(),
                    lr=learning_rate,
                    weight_decay =5e-5)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//2, eta_min=1e-8)
    
    def val(optimizer,snn,test_loader,test_dataset,batch_size,epoch):
        print('===> evaluating models...')
        snn.eval()
        correct = 0
        total = 0
        predicted = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                if ((batch_idx+1)<=len(test_dataset)//batch_size):
                    optimizer.zero_grad()
                    outputs = snn(inputs.to(device))
                    labels = labels.squeeze(1)
                    _ , predicted = outputs.cpu().max(1)
                    total += float(labels.size(0))
                    correct += float(predicted.eq(labels).sum())
        acc = 100. * float(correct) / float(total)
        print('================')
        print('val acc:',acc , '%  epoch:',epoch)
        print('================')
        writer.add_scalar('acc', acc, epoch)
        return acc

    save_folder = './saved_model'

    for epoch in range(num_epochs):
        if epoch>1:
            for p in snn.parameters():
                p.requires_grad=True #TODO
        snn.train()
        running_loss = 0
        start_time = time.time() 
        print('===> training models...')
        correct = 0.0
        total = 0.0
        torch.cuda.empty_cache()
        for i, (images, labels) in enumerate(train_loader):
            if ((i+1)<=len(train_dataset)//batch_size):
                with autocast():
                    outputs = snn(images.to(device)).cpu()
                    labels = labels.squeeze(1)
                    _ , predict = outputs.max(1)
                    loss = criterion(outputs, labels)
                    correct += predict.eq(labels).sum()
                    total += float(predict.size(0))

                    loss /= accumulation
                    running_loss += loss.item()
                    loss.backward()

                if (i+1)%accumulation == 0:
                    optimizer.step()
                    snn.zero_grad()
                    optimizer.zero_grad()

                if (i+1)%(10) == 0:
                    if not os.path.isdir(save_folder):
                        os.mkdir(save_folder)
                    train_acc =  100* correct / total
                    print('=====> Epoch [%d/%d], Step [%d/%d], Loss: %.5f , Acc: %.5f  '
                      %(epoch+start_epoch, num_epochs+start_epoch, i+1, len(train_dataset)//(batch_size),running_loss, train_acc))
                    writer.add_scalar('running Loss', running_loss, training_iter)
                    writer.add_scalar('running Acc', train_acc, training_iter)
                    correct = 0.0
                    total = 0.0
                    running_loss = 0
                    #
            training_iter +=1 
        torch.cuda.empty_cache()
        #evaluation
        acc = val(optimizer,snn,test_loader,test_dataset,batch_size,epoch)
        lr_scheduler.step()
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        if acc > bestacc:
            bestacc = acc
            print('===> Saving models...')


    torch.save(snn.state_dict(),'./'+save_folder+'/'+task+str(int(bestacc))+'.pkl')