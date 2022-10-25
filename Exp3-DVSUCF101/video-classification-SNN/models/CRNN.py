import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# CRNN module
def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


# 2D CNN encoder train from scratch (no transfer learning)
class EncoderCNN(nn.Module):
    def __init__(self, img_h=90, img_w=120, fc_hidden=512, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        # self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.ch1, self.ch2, self.ch3, self.ch4 = 8, 16, 32, 64
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_h, self.img_w), self.pd1, self.k1,
                                                 self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden = fc_hidden
        #self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3,
                      padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4,
                      padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        # self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1],
                             self.fc_hidden)  # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden, self.CNN_embed_dim)  # output = CNN embedding latent variables
        #self.fc = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)  # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            #x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc2(x)

            #x = self.fc(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


# 2D CNN encoder using ResNet (Original version)
class EncoderResCNN_origin(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, CNN_embed_dim=300):
        """Load the pretrained ResNet and replace top fc layer."""
        super(EncoderResCNN_origin, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        # self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


# 2D CNN encoder using ResNet (mashijie version)
class EncoderResCNN(nn.Module):
    def __init__(self, fc_hidden=1024, CNN_embed_dim=300):
        """Load the pretrained ResNet and replace top fc layer."""
        super(EncoderResCNN, self).__init__()

        self.fc_hidden = fc_hidden

        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        # self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden)
        # print(resnet.fc.in_features)   # resnet18: 512
        # self.bn1 = nn.BatchNorm1d(fc_hidden, momentum=0.01)
        # self.fc2 = nn.Linear(fc_hidden, CNN_embed_dim)

        self.fc = nn.Linear(resnet.fc.in_features, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
            x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.fc(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, num_layers=2, hidden_dim=256, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.num_layers = num_layers  # RNN hidden layers
        self.hidden_dim = hidden_dim  # RNN hidden nodes
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, c_n) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), c_n shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc(RNN_out[:, -1, :])  # choose RNN_out at the last time step

        return x


# average RNN decoder over time-steps before fed into fc_layer
class DecoderRNN_avg(nn.Module):
    def __init__(self, CNN_embed_dim=300, num_layers=2, hidden_dim=256, num_classes=50):
        super(DecoderRNN_avg, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.num_layers = num_layers  # RNN hidden layers
        self.hidden_dim = hidden_dim  # RNN hidden nodes
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, c_n) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), c_n shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        RNN_out = torch.mean(RNN_out, 1)
        x = self.fc(RNN_out)  # choose RNN_out at the last time step

        return x
