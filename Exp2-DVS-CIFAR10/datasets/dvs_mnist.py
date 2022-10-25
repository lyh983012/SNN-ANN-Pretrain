import scipy.io as sio
import os 
import numpy as np
import torch
import linecache
import torch.utils.data as data

rate = 0.8

class DVS_MNIST_Dataset(data.Dataset):
    def __init__(self, mode, scale = 4,data_set_path='/data/MNIST_DVS_mat/', timestep = 50):
        super().__init__()
        print('using  2s recording ')
        self.mode = mode
        self.train_filenames = []
        self.test_filenames = []
        
        self.scale=scale
        scale_folder = '/scale'+str(scale)
        
        for folder in os.listdir(data_set_path):
            label  = folder[-1]
            path_prefix = data_set_path+folder+scale_folder+'/'
            count = 0 
            for file in os.listdir(path_prefix):
                count += 1
                if count <= int(1000*rate):
                    self.train_filenames.append((path_prefix+file,label))
                else:
                    self.test_filenames.append((path_prefix+file,label))    
        self.timestep=timestep#time resolution,us

    def __getitem__(self, index):
        if self.mode == 'train':
            info = self.train_filenames
        else:
            info = self.test_filenames
        filename, classnum = info[index]

        roadef_info = sio.loadmat(filename)
        data = roadef_info['dd']
        classnum = int(classnum)
        maxtime =  data[-1,0]
        tensor_data = torch.zeros([2,self.timestep,128,128])
        time_resolution =  data[-1,0]//self.timestep
        
        for i in range(self.timestep):
            data_slide = data[i*time_resolution:(i+1)*time_resolution,:]
            ones  = data_slide[data_slide[:,-1]==1]
            minusones  =  data_slide[data_slide[:,-1]==-1]
            
            tensor_data[0,i,ones[:,3],ones[:,4]]=1
            tensor_data[1,i,minusones[:,3],minusones[:,4]]=1
            
        return tensor_data,torch.tensor(classnum)


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_filenames)
        else:
            return len(self.test_filenames)
        
    
    
test_dataset = DVS_MNIST_Dataset(mode='train',scale = 4,data_set_path='/data/MNIST_DVS_mat/',timestep = 50)

for data,label in test_dataset:
    #print(data,label)
    pass