#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 10:24:08 2019

@author: liang
"""

# postprocess
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

predict_dir = './predicts'
if not os.path.exists(predict_dir):
  os.makedirs(predict_dir)

class2value = {0:0,1:20,2:60,3:40}

def postprocess(num=1):
  # switch region
  rows = 2544
  cols = 7194
  filenames = os.listdir('./results/region')
  filepaths = [os.path.join('./results/region',name) for name in filenames]
  Region = np.zeros(shape=(rows*7,cols*7),dtype=np.int16)
  for filepath in tqdm(filepaths,desc='Switching Region'):
    data = np.load(filepath)
    coord = data['coord']
    predict = data['predict']
    predict_value = [class2value[label] for label in predict]
    x1,x2 = coord[:,0]*7,(coord[:,0]+1)*7
    y1,y2 = coord[:,1]*7,(coord[:,1]+1)*7
    for idx1,idx2,idy1,idy2,value in zip(x1,x2,y1,y2,predict_value):
      Region[idx1:idx2,idy1:idy2] = value
  
  # switch column margin
  col_margins = np.zeros(shape=(rows*7,4),dtype=np.int16)
  row_margins = np.zeros(shape=(2,cols*7),dtype=np.int16)
  margin_names = os.listdir('./results/margin')
  margin_paths = [os.path.join('./results/margin',name) for name in margin_names]
  for path in tqdm(margin_paths,desc='Switch Margin'):
    data = np.load(path)
    coord = data['coord']
    predict = data['predict']
    predict_value = [class2value[label] for label in predict]
    for (x1,y1),value in zip(coord,predict_value):
      # row margin
      if x1==-1:
        row_margins[:,y1*7:(y1+1)*7] = value
      else:
      # col margin
        col_margins[x1*7:(x1+1)*7,:] = value
        
  # right down little patch
  patch = np.ones(shape=[2,4],dtype=np.int16)*row_margins[1,-1]
  
  # switch a full image to submit
  row_margins = np.hstack((row_margins,patch))
  # merge col margins
  Region = np.hstack((Region,col_margins))
  # merge row margins
  Region = np.vstack((Region,row_margins))
  assert Region.shape==(17810,50362)
  
  # fusion train pixel
  train_infos = np.load('./train_infos.npy').astype(np.int16)
  for info in tqdm(train_infos,desc='Merge Training Data'):
    label,center_x,center_y = info
    value = class2value[label]
    x1,x2 = center_x-3,center_x+4
    y1,y2 = abs(center_y)-3,abs(center_y)+4
    Region[y1:y2,x1:x2] = value
  
  # convert to pil image
  prediction = Image.fromarray(Region)
  prediction.save(os.path.join(predict_dir,'%d.tif'%num))
  
  
if __name__ == '__main__':
  postprocess(num=1)
