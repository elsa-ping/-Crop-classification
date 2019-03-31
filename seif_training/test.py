#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:43:18 2019

@author: liang
"""

import torch
from torch.nn import functional as F
from model import TinyNet
from dataset import TestLoader
from tqdm import tqdm
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#===========================
# data_type='margin'
data_type='margion'
#=================================

result_dir = os.path.join('./results',data_type)

if not os.path.exists(result_dir):
  os.makedirs(result_dir)


def test():
  """
  return batchx4 numpy array
  row_index,col_index,predicts,probs
  """
  # config
  batchsize=2048
  # model
  tinynet = torch.nn.DataParallel(TinyNet().cuda())
  #=====================================
  # ckpt = './checkpoints1/Num-1.pth'
  ckpt = './checkpoints1/Num-1.pth'
  #========================================
  # load parameters
  tinynet.load_state_dict(torch.load(ckpt))
  testloader = TestLoader(data_type,batchsize,sampling=False)
  # evaluation
  tinynet.eval()
  
  # testing
  with torch.no_grad():
    for idx,(image,coord) in enumerate(tqdm(testloader,desc='Testing!!!')):
      image = image.cuda()
      output = tinynet(image)
      output = F.softmax(output,dim=-1)
      # ----------------------# 
      probs,predicts = torch.max(output,dim=-1)
      probs = probs.cpu().numpy()
      predicts = predicts.cpu().numpy()
      mask = probs<0.9
      predicts[mask] = 0
      coord = coord.numpy()
      np.savez(os.path.join(result_dir,'%d.npz'%(idx+1)),coord=coord,predict=predicts,prob=probs)
      
  print('\nTest Finished!!!')


if __name__ == '__main__':
  test()  
      
  

