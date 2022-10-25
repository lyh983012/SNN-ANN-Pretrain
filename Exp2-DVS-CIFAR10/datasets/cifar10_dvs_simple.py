import torch
import numpy as np
import torch.utils.data as data
import scipy.io as sio 
import h5py

class cifar10_DVS(data.Dataset):
    def __init__(self, path='load_test.mat',method = 'h',wins = 12):
        if method=='h':
            data = h5py.File(path)
            image,label = data['image'],data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image[:,:,:,:,:wins]).float()
            self.labels = torch.from_numpy(label).float()
        else:
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image']).float()
            self.labels = torch.from_numpy(data['label']).float()
            self.images = self.images[:,:,:,:,:wins]


        self.num_sample = int((len(self.images)//100)*100)
        print(self.images.size(),self.labels.size())

    def __getitem__(self, index):#返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample