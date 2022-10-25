"""
train LIAF from pretrained ANN models
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
from utils import labels2cat
from dataloader import DVS_Dataset

from configs_LIAF import training_configs, model_configs

import sys
sys.path.append(r'./models/')   # caution, please note to add models path
from models.LIAFCNN import LIAFCNN
from models.CRNN import DecoderRNN
from models import LIAF


# set path
data_path = r'/data1/sjma/UCF101/jpegs_256/'    # define UCF-101 RGB data path
events_path = r'/data1/sjma/UCF101/events_dvs/'    # define UCF-101 events data path
action_name_path = r'/data1/sjma/UCF101/UCF101actions.pkl'
ckpts_path = r'./checkpoints/LIAF_from_CNN/'
tensorboard_path = r'./tensorboard/LIAF_from_CNN/'
dic_save_path = r'./checkpoints/LIAF_from_CNN/configs_LIAF.txt'
log_path = r'./log/LIAF_from_CNN/'

if not os.path.exists(ckpts_path):
    os.mkdir(ckpts_path)
if not os.path.exists(tensorboard_path):
    os.mkdir(tensorboard_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)

# device settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}


# save configs and hyper-params
model_configs['load_path'] = r'./checkpoints/ANN_t15/ann-epoch47-best.pth.tar'   # note: load checkpoints path
f_0 = open(dic_save_path, 'w')
f_0.write(str(model_configs))
f_0.write('\n')
f_0.write(str(training_configs))
f_0.close()


# preprocess for dataset
# ==============================================================================================
# ==============================================================================================
# load UCF101 actions names
with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

actions = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f[(loc1 + 2): loc2])

    all_names.append(f)

# list all data files
all_X_list = all_names                  # all video file names
all_y_list = labels2cat(le, actions)    # all video labels


# dataset and model
# ==============================================================================================
# ==============================================================================================
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list,
                                                                  test_size=0.25, random_state=42)

train_transform = transforms.Compose([
    transforms.Resize(model_configs['encoder']['input_size']),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(model_configs['encoder']['input_size']),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

selected_frames = np.arange(training_configs['begin_frame'], training_configs['end_frame'],
                            training_configs['skip_frame']).tolist()

train_set = DVS_Dataset(events_path, train_list, train_label, selected_frames, transform=train_transform)
test_set = DVS_Dataset(events_path, test_list, test_label, selected_frames, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=training_configs['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=training_configs['batch_size'], shuffle=False, **kwargs)


def train(configs, model, device, train_loader, optimizer, criterion, epoch, writer, f_log, running_loss_list):
    print('\n\nEpoch: %d' % epoch)
    print("=====================================================================")
    print("Begin Training!")
    start_time = time.time()
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()
    running_loss = 0.0
    correct = 0
    for batch_idx, (events, labels) in enumerate(train_loader):
        events, labels = events.to(device), labels.to(device).view(-1, )
        optimizer.zero_grad()

        # (B, T, C, H, W) -> (B, C, T, H, W)
        events = events.float()
        events = events.permute(0, 2, 1, 3, 4)
        features = cnn_encoder(events)

        outputs = rnn_decoder(features)
        loss = criterion(outputs, labels)

        writer.add_scalar('running_loss', loss, configs['steps'])
        loss.backward()
        optimizer.step()
        configs['steps'] += 1

        running_loss += loss.item()

        # predict
        preds = torch.max(outputs, 1)[1]  # y_pred != output
        correct += preds.eq(labels.view_as(preds)).sum().item()

        if batch_idx % configs['log_interval'] == configs['log_interval'] - 1 or batch_idx == len(train_loader) - 1:
            if batch_idx == len(train_loader) - 1 and len(train_loader) % configs['log_interval'] != 0:
                running_loss *= configs['log_interval'] /\
                                (len(train_loader) - len(train_loader) //
                                 configs['log_interval'] * configs['log_interval'])
            print('Epoch [%3d/%3d], Step [%3d/%3d], running_loss: %.06f'
                  % (epoch, configs['total_epoch'], batch_idx + 1, len(train_loader),
                     running_loss / configs['log_interval']))
            elapsed_time = time.time() - start_time
            print('Time elapsed: %.5f' % elapsed_time)

            f_log.write('Epoch [%3d/%3d], Step [%3d/%3d], running_loss: %.06f'
                        % (epoch, configs['total_epoch'], batch_idx + 1, len(train_loader),
                            running_loss / configs['log_interval']))
            f_log.write('\n')
            f_log.write('Time elapsed: %.5f' % elapsed_time)
            f_log.write('\n')

            running_loss_list.append(running_loss / configs['log_interval'])
            running_loss = 0.0

    print('Training: Accuracy: {}/{} ({}%)'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    train_accuracy = correct / len(train_loader.dataset)

    f_log.write('Training: Accuracy: {}/{} ({}%)'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    f_log.write('\n')

    return train_accuracy


def eval_test(model, device, criterion, test_loader, f_log):
    print("\n=====================================================================")
    print("Begin Evaluation!")
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for events, labels in test_loader:
            events, labels = events.to(device), labels.to(device).view(-1, )

            # (B, T, C, H, W) -> (B, C, T, H, W)
            events = events.float()
            events = events.permute(0, 2, 1, 3, 4)
            features = cnn_encoder(events)

            outputs = rnn_decoder(features)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # predict
            preds = torch.max(outputs, 1)[1]  # y_pred != output
            correct += preds.eq(labels.view_as(preds)).sum().item()

    test_loss /= len(test_loader)
    print('test_loss: %.06f' % test_loss)
    f_log.write('test_loss: %.06f' % test_loss)
    f_log.write('\n')

    print('Test: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    f_log.write('Test: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    f_log.write('\n')
    test_accuracy = correct / len(test_loader.dataset)

    return test_accuracy, test_loss


def main():
    # init model
    # model_configs['encoder']['cfg_cnn'][0] = (3, 8, (5, 5), (2, 2), (0, 0), False, 2, 2)
    cnn_encoder = LIAFCNN(model_configs['encoder']).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=model_configs['encoder']['embed_dim'],
                             num_layers=model_configs['decoder']['num_layers'],
                             hidden_dim=model_configs['decoder']['hidden_dim'],
                             num_classes=model_configs['decoder']['num_classes']).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn_encoder = nn.DataParallel(cnn_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)

    # load from checkpoints (pretrained CRNN)
    # ===================================================================================================
    ckpts = torch.load(model_configs['load_path'])
    cnn_dict = cnn_encoder.state_dict()
    pretrained_cnn_dict = ckpts['cnn_encoder_state_dict']
    new_dict = {k: v for k, v in pretrained_cnn_dict.items() if 'module.network.cnn0' not in k}
    cnn_dict.update(new_dict)
    cnn_encoder.load_state_dict(cnn_dict)
    # ===================================================================================================

    crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=training_configs['lr'])
    # optimizer.load_state_dict(ckpts['optimizer_state_dict'])  # note!!!
    criterion = nn.CrossEntropyLoss()

    # init tensorboard and list
    writer = SummaryWriter(tensorboard_path)
    train_acc_list = []
    test_acc_list = []
    running_loss_list = []
    test_loss_list = []
    best_test_acc = 0.0

    print("=====================================================================")
    print("=====================================================================")
    print("Start Training!\n")
    with open(os.path.join(log_path, "LIAF_screen_log.txt"), "w") as f1:
        for epoch in range(1, training_configs['total_epoch'] + 1):
            start_time = time.time()
            # adjust learning rate for SGD
            # adjust_learning_rate(optimizer, epoch)

            # train
            train_acc = train(configs=training_configs, model=[cnn_encoder, rnn_decoder], device=device,
                              train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=epoch,
                              writer=writer, f_log=f1, running_loss_list=running_loss_list)

            # evaluation
            test_acc, test_loss = eval_test(model=[cnn_encoder, rnn_decoder], device=device, criterion=criterion,
                                            test_loader=test_loader, f_log=f1)

            writer.add_scalar('accuracy/train', train_acc, epoch)
            writer.add_scalar('accuracy/test', test_acc, epoch)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)

            # record time
            elapsed_time = time.time() - start_time
            print('Total time elapsed: %.5f' % elapsed_time)
            f1.write('Total time elapsed: %.5f' % elapsed_time)
            f1.write('\n\n\n')
            f1.flush()

            if test_acc > best_test_acc:
                print("Congrats, best results!")
                f2 = open(os.path.join(log_path, "LIAF_best_test_acc.txt"), "w")
                f2.write("EPOCH = %d, best_test_acc = %.03f%%" % (epoch, 100 * test_acc))
                f2.close()
                best_test_acc = test_acc

                if epoch >= training_configs['total_epoch'] / 2:
                    print('Saving checkpoints......')
                    checkpoints = {'cnn_encoder_state_dict': cnn_encoder.state_dict(),
                                   'rnn_decoder_state_dict': rnn_decoder.state_dict(),
                                   'optimizer_state_dict': optimizer.state_dict(),
                                   'epoch': epoch,
                                   'train_acc_list': train_acc_list,
                                   'test_acc_list': test_acc_list,
                                   'running_loss_list': running_loss_list,
                                   'test_loss_list': test_loss_list,
                                   'best_test_acc': best_test_acc
                                   }

                    torch.save(checkpoints, os.path.join(ckpts_path, 'liaf-epoch{}-best.pth.tar'.format(epoch)))

            # save checkpoints
            if epoch % training_configs['save_interval'] == 0 or epoch == training_configs['total_epoch']:
                # torch.save(model.state_dict(),
                #            os.path.join(model_dir, 'model-epoch{}.pt'.format(epoch)))
                print('Saving checkpoints......')
                checkpoints = {'cnn_encoder_state_dict': cnn_encoder.state_dict(),
                               'rnn_decoder_state_dict': rnn_decoder.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict(),
                               'epoch': epoch,
                               'train_acc_list': train_acc_list,
                               'test_acc_list': test_acc_list,
                               'running_loss_list': running_loss_list,
                               'test_loss_list': test_loss_list,
                               'best_test_acc': best_test_acc
                               }

                torch.save(checkpoints, os.path.join(ckpts_path, 'liaf-epoch{}.pth.tar'.format(epoch)))

    writer.close()

    f3 = open(os.path.join(log_path, "LIAF_record_lists.txt"), "w")
    f3.write("running_loss_list: \n")
    f3.write(str(running_loss_list))
    f3.write("\n")
    f3.write("test_loss_list: \n")
    f3.write(str(test_loss_list))
    f3.write("\n")
    f3.write("train_acc_list: \n")
    f3.write(str(train_acc_list))
    f3.write("\n")
    f3.write("test_acc_list: \n")
    f3.write(str(test_acc_list))
    f3.write("\n")
    f3.close()


if __name__ == '__main__':
    main()
