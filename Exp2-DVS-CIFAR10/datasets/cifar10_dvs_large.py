import torch
import numpy as np
import torch.utils.data as data
import scipy.io as sio 
import os

class cifar10_DVS(data.Dataset):
    def __init__(self, path='/data/CIFAR10-MAT',mode = 'train',wins = 12):
        self.mode = mode
        self.filenames = []
        self.trainpath = path+'/train/'
        self.testpath = path+'/test/'
        self.formats = '.mat'
        self.wins = wins
        if mode == 'train':
            self.path = self.trainpath
            for file in os.listdir(self.trainpath):
                self.filenames.append(self.trainpath+file)
        else:
            self.path = self.trainpath
            for file in os.listdir(self.testpath):
                self.filenames.append(self.testpath+file)



        self.num_sample = int(len(self.filenames))

    def __getitem__(self, index):#返回的是tensor
        image = 0
        label = 0
        try:
            data = sio.loadmat(self.filenames[index])
            image,label,name = data['frame_Data'],data['label'],data['name']
            image = image.astype(np.float32)
            label = label.astype(np.float32)
            image = torch.from_numpy(image[:,:self.wins,:,:]).float()
            label = torch.from_numpy(label[0,:]).long()-5
        except:
            image = torch.zeros([2,8,128,128]).float()
            label = torch.zeros([1]).long()
        return image,label

    def __len__(self):
        return self.num_sample