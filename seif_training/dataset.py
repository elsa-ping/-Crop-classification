#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:00:12 2019

@author: ping
"""

# load data
import torch
from torch.utils import data
import numpy as np
import os
import random
import glob


class TrainDataset(data.Dataset):
  
  def __init__(self):
    super(TrainDataset,self).__init__()
		#==========================================    
		#self.root_path = './dataset/train'
    self.root_path = './dataset_1/train' #self_training_2.py
		#========================================== 
    self.file_names = os.listdir(self.root_path)
    self.file_paths = [os.path.join(self.root_path,name) for name in self.file_names]
    
  def __len__(self):
    return len(self.file_paths)
  
  
  def __getitem__(self,idx):
    file_path = self.file_paths[idx]
    # load data
    data = np.load(file_path)
    image = data['image']
    label = data['label']
    # normalize image
    mean,std = np.mean(image,axis=(0,),keepdims=True),np.std(image,axis=(0,),keepdims=True)
    image = (image-mean)/(std+1e-12)
    # random data augment
    # random rorate
    idx = np.random.randint(0,6)
    if idx==0:
      image = np.rot90(image,k=1,axes=(1,2))
    elif idx==2:
      image = np.rot90(image,k=2,axes=(1,2))
    elif idx==4:
      image = np.rot90(image,k=3,axes=(1,2))
    # random flip
    if np.random.uniform()>0.5:
      # random flip updown
      image = image[:,::-1,:]
    if np.random.uniform()>0.5:
      # random flip left-right
      image = image[:,:,::-1]
    # convert torch tensor
    # copy is very important for pytorch contigous
    image = torch.from_numpy(image.copy()).float()
    label = torch.from_numpy(label).long()
    return image,label


class TestDataset(data.Dataset):
  
  def __init__(self,data_type='region',sampling=True):
    super(TestDataset,self).__init__()
    self.data_type = data_type		
		#================================================		
		#self.data_names = glob.glob(os.path.join('./dataset/test',self.data_type,'*.npz'))
		#================================================		
    self.data_names = glob.glob(os.path.join('./dataset_1/test',self.data_type,'*.npz'))  #self_training_2.py
    self.data_infos = np.load(self.data_names[0])
    self.images = self.data_infos['data']
    self.coords = self.data_infos['coord']
    self.nums = len(self.images)
    
    if sampling:
      indexes = np.random.randint(0,self.nums,size=100000)
      self.test_imgs, self.test_coords = self.images[indexes],self.coords[indexes]
    else:
      self.test_imgs, self.test_coords = self.images,self.coords
  
  def __len__(self):
    return len(self.test_imgs)
  
  def __getitem__(self,idx):
    image = self.test_imgs[idx]
    coord = self.test_coords[idx]
    # simple normalization
    # every pixel along depth
    mean,std = np.mean(image,axis=(0,),keepdims=True),np.std(image,axis=(0,),keepdims=True)
    image = (image-mean)/(std+1e-12)
    # convert to pytorch tensor
    image = torch.from_numpy(image).float()
    coord = torch.from_numpy(coord).int()
    return image,coord
    
    
def TrainLoader(batchsize):
  traindataset = TrainDataset()
  trainloader = data.DataLoader(traindataset,
                                batch_size=batchsize,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True)
  return trainloader


def TestLoader(data_type='region',batchsize=8192,sampling=True):
  testdataset = TestDataset(data_type,sampling)
  testloader = data.DataLoader(testdataset,
                               batch_size=batchsize,
                               num_workers=16,
                               pin_memory=True)
  return testloader

if __name__ == '__main__':
  from tqdm import tqdm
  # trainloader = TrainLoader(256)
  testloader = TrainLoader(2048)
  pbar = tqdm(testloader)
  for img,coord in testloader:
    print(img.shape)
    print(coord.shape)
    
  
    

    
    
    
