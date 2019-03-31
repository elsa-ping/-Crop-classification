#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:43:18 2019

@author: liang
"""

import torch
from torch import optim,nn
from model2 import TinyNet
from dataset import TrainLoader,TestLoader
from tqdm import tqdm
import os
import shutil
import numpy as np
import glob
from sklearn.externals import joblib


# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

#===============================
#checkpoints = './checkpoints1'
checkpoints = './checkpoints2'
#===============================
if not os.path.exists(checkpoints):
  os.makedirs(checkpoints)

def train():
  # config
  recurrents = 10
  epochs=20
  batchsize=256 * 2
  # model
  tinynet = nn.DataParallel(TinyNet().cuda())
  # optimizier
  optimizer = optim.Adam(tinynet.parameters(),
                         betas=(0.5,0.999),
                         weight_decay=1e-4)
  
  # loss function
  loss_func = nn.CrossEntropyLoss().cuda()
  # loading parameters
  #tinynet.load_state_dict(torch.load('./checkpoints1/Num-10.pth'))
  
  for self_num in range(0,recurrents):
    trainloader = TrainLoader(batchsize)
    print('\033[1;32m----------------Self-Training Nums:%d----------------\033[0m'%(self_num+1))
    tinynet.train()
    for epoch in range(epochs):
      # compute epoch accuracy 
      num_corrects = 0
      num_total = 0
      epoch_loss = []
      pbar = tqdm(trainloader)
      for image,label in pbar:
        image = image.cuda()
        label = label.cuda()
        output = tinynet(image)
        # compute loss
        loss = loss_func(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ----------------------#  dimension 2
        predicts = torch.argmax(output,dim=-1)
        corrects = (predicts==label).sum()
        batch_acc = corrects.float()/label.size(0)
        num_corrects += corrects
        num_total += label.size(0)
        epoch_loss.append(loss.item())
        fmt = 'Epoch[{:2d}]-Loss:{:.3f}-Batch_acc:{:.3f}'.format(epoch+1,loss.item(),batch_acc.item())
        pbar.set_description(fmt)
        
      epoch_accu = num_corrects.float()/num_total
      avg_loss = sum(epoch_loss)/len(epoch_loss)
      print('\033[1;31mNums:[%2d/%2d]-Epoch:%2d-Accu:%.3f-Loss:%.3f\033[0m'\
                  %(self_num+1,recurrents,epoch+1,epoch_accu.item(),avg_loss))
  
    # every recurrent save model
    torch.save(tinynet.state_dict(),os.path.join(checkpoints,'Num-%d.pth'%(self_num+1)))
    # test for generate fake label
    tinynet.eval()
    # delete original train dataset
    shutil.rmtree('./dataset_1/train')
    # make a new training dataset
    train_dir = './dataset_1/train'
    os.makedirs(train_dir)
    # using labeled data
    files = glob.glob('./dataset_1/train_1/*.npz')
    joblib.Parallel(n_jobs=16)\
    (joblib.delayed(copying_file)(train_dir,file) for file in tqdm(files,desc='Copying Labeled Training Data'))
    
    with torch.no_grad():
      count = 0
      testloader = TestLoader(batchsize=2048,sampling=True)
      for image,coord in tqdm(testloader,desc='Generating Fake Labels'):
        image = image.cuda()
        output = tinynet(image)
        output = nn.functional.softmax(output,dim=-1)
        probs,predicts = torch.max(output,dim=-1)
        mask = probs>0.9
        imgs = image[mask].cpu().numpy()
        predicts = predicts[mask].cpu().numpy()
        for img,lab in zip(imgs,predicts):
          count += 1
          np.savez(os.path.join(train_dir,'%d.npz'%count),image=img,label=lab)
        
  print('Train Finished!!!')



def copying_file(train_dir,file):
    shutil.copy(file,train_dir,follow_symlinks=False)

 
if __name__ == '__main__':
  train()  
      
  
