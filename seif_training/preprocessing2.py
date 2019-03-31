#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:29:09 2019

@author: liang
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class2idx = {'玉米':1,'水稻':2,'大豆':3}


def load_info():
  """
  load information from train.txt
  """
  infos = [] # every element is a tuple (cls,x,y)
  filepath = './train.txt'
  # load coordinates
  coordinates = np.loadtxt(filepath,skiprows=1,delimiter=',', usecols=[5,6])
  # load class
  labels = np.loadtxt(filepath,dtype=str,skiprows=1,usecols=[2],delimiter=',')
  
  # convert to idx
  labels = [class2idx[i] for i in labels]
  infos = np.array([(label,round(x),round(y)) for label,(x,y) in zip(labels,coordinates)],dtype=np.int64)
  
  # load some bg datas
  background1 = np.load('./bg_coord_1.npy').astype(np.int64)
  bg1_label = np.zeros(shape=(len(background1),1),dtype=np.int64)
  bg1_infos = np.hstack((bg1_label,background1))
  
  infos = np.vstack((infos,bg1_infos))

  np.save('./train_infos.npy',infos) 
  return infos


def extract_train_patches():
  infos = np.load('./train_infos.npy').astype(np.int32)
  # load image
  image = io.imread('./image.tif')
  datadir = './dataset_1/train_1'		#更改了保存位置
  
  if not os.path.exists(datadir):
    os.makedirs(datadir)
    
  for idx,info in enumerate(tqdm(infos,desc='Generating Train Patches')):
    label,center_x,center_y = info
    # crop patch
    x1,x2 = center_x-16,center_x+17   # crop 33x33  original 7x7		#patch 由 7*7改为32*32
    y1,y2 = abs(center_y)-16,abs(center_y)+17
    patch = image[:,y1:y2,x1:x2]
    np.savez(os.path.join(datadir,'true_%d.npz'%(idx+1)),image=patch,label=label)
    


def make_train_background():
  datadir = './dataset/train'
  
  if not os.path.exists(datadir):
    os.makedirs(datadir)
    
  image = io.imread('./image.tif')
  x1 = np.arange(0,5000,7,dtype=np.int16)
  y1 = np.arange(0,5000,7,dtype=np.int16)
  x2 = np.arange(15835,15935,7,dtype=np.int16)
  y2 = np.arange(3867,3918,7,dtype=np.int16)
  x3 = np.arange(45000,50362,7,dtype=np.uint16)
  y3 = np.arange(15000,17810,7,dtype=np.uint16)
  count = 0
  label = 0
  for center_x in tqdm(x1):
    for center_y in y1:
      count += 1
      patch = image[:,center_y:center_y+7,center_x:center_x+7]
      if patch.shape==(8,7,7):
        np.savez(os.path.join(datadir,'bg-%d.npz'%(count)),image=patch,label=label)
  
  for center_x in tqdm(x2):
    for center_y in y2:
      count += 1
      patch = image[:,center_y:center_y+7,center_x:center_x+7]
      if patch.shape==(8,7,7):
        np.savez(os.path.join(datadir,'bg-%d.npz'%(count)),image=patch,label=label)
      
  for center_x in tqdm(x3):
    for center_y in y3:
      count += 1
      patch = image[:,center_y:center_y+7,center_x:center_x+7]
      if patch.shape==(8,7,7):
        np.savez(os.path.join(datadir,'bg-%d.npz'%(count)),image=patch,label=label)


def extract_test_patches():
  """
  return patch: coord:(row,col),image:8x7x7 patch
  maxium: row=2544, col=7194
  """
  image = io.imread('./image.tif')
  # save every patch
  datadir1 = './dataset_1/test/region'		#更改了位置
  
  if not os.path.exists(datadir1):
    os.makedirs(datadir1)
  row_max = 556          # row_max=2544  7x7    32×32	17810/32
  col_max = 1573         # col_max=7194			50362/32
  count = 0		
  test_datas = np.zeros(shape=[row_max*col_max,8,32,32],dtype=np.int16)			#将7*7改为32*32
  test_coords = np.zeros(shape=[row_max*col_max,2],dtype=np.int16)
  for row_x in tqdm(range(row_max),desc='Generating test patches!!!'):
    # compute row number
    x1,x2 = 32*row_x,32*(row_x+1)
    for col_y in range(col_max):
      # compute column number
      y1,y2 = 32*col_y,32*(col_y+1)
      # crop patch
      patch = image[:,x1:x2,y1:y2]
      test_datas[count] = patch
      test_coords[count] = np.array([row_x,col_y],dtype=np.int16)
      count += 1
  np.savez(os.path.join(datadir1,'region.npz'),data=test_datas,coord=test_coords)
  

def extract_margin_test_patches():     
  # simple save margin pixels
  datadir2 = './dataset_1/test/margin'			#改变了位置
  if not os.path.exists(datadir2):
    os.makedirs(datadir2)
    
  image = io.imread('./image.tif')
  
  col_margin = image[:,:-18,-32:]			#改变了大小
  assert col_margin.shape==(8,17792,32)
  row_margin = image[:,-32:,:-26]
  assert row_margin.shape==(8,32,50336)
  
  row_max = 556
  col_max = 1573
  
  columns = np.zeros(shape=[row_max,8,32,32],dtype=np.int16)
  columns_coords = np.zeros(shape=[row_max,2],dtype=np.int16)
  rows = np.zeros(shape=[col_max,8,32,32],dtype=np.int16)
  rows_coords = np.zeros(shape=[col_max,2],dtype=np.int16)
  
  for idx,col in enumerate(range(col_max)):
    x1,x2 = 32*col,32*(col+1)
    patch = row_margin[...,x1:x2]
    rows[idx] = patch
    rows_coords[idx] = np.array([-1,col],dtype=np.int16)
  # np.savez(os.path.join(datadir2,'rows.npz'),data=rows,coord=rows_coords)
  
  for idx,row in enumerate(range(row_max)):
    x1,x2 = 32*row,32*(row+1)
    patch = col_margin[:,x1:x2,:]
    columns[idx] = patch
    columns_coords[idx] = np.array([row,-1],dtype=np.int16)

  margins = np.vstack((columns,rows))
  margins_coord = np.vstack((columns_coords,rows_coords))
  
  np.savez(os.path.join(datadir2,'margin.npz'),data=margins,coord=margins_coord)
  
if __name__ == '__main__': 
  # load_info()
  # extract_train_patches()
  extract_margin_test_patches()
  extract_test_patches()
