#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:40:45 2019

@author: wsw
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
  # datadir = './dataset/train_1'
  datadir = './dataset/train'
 
  
  
  if not os.path.exists(datadir):
    os.makedirs(datadir)
    
  for idx,info in enumerate(tqdm(infos,desc='Generating Train Patches')):
    label,center_x,center_y = info
    # crop patch
    x1,x2 = center_x-3,center_x+4   # crop 33x33  original 7x7
    y1,y2 = abs(center_y)-3,abs(center_y)+4
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
  datadir1 = './dataset/test/region'
  
  if not os.path.exists(datadir1):
    os.makedirs(datadir1)
  row_max = 2544          # row_max=2544  7x7 17810/7
  col_max = 7194         # col_max=7194		50362/7
  count = 0
  """
  for row_x in enumerate(tqdm(range(row_max),desc='Generating test patches!!!')):		#保存每一行/列的数据
    # compute row number
    x1,x2 = 7*row_x,7*(row_x+1)
    for col_y in range(col_max):
      # compute column number
      y1,y2 = 7*col_y,7*(col_y+1)
      # crop patch
      patch = image[:,x1:x2,y1:y2]
      count += 1
      np.savez(os.path.join(datadir1,'%d.npz'%count),image=patch,coord=(row_x,col_y))			
  """
  test_datas = np.zeros(shape=[row_max*col_max,8,7,7],dtype=np.int16)				#所有数据保存为一个npz文件
  test_coords = np.zeros(shape=[row_max*col_max,2],dtype=np.int16)
  for row_x in tqdm(range(row_max),desc='Generating test patches!!!'):
    # compute row number
    x1,x2 = 7*row_x,7*(row_x+1)
    for col_y in range(col_max):
      # compute column number
      y1,y2 = 7*col_y,7*(col_y+1)
      # crop patch
      patch = image[:,x1:x2,y1:y2]
      test_datas[count] = patch
      test_coords[count] = np.array([row_x,col_y],dtype=np.int16)
      count += 1
  np.savez(os.path.join(datadir1,'region.npz'),data=test_datas,coord=test_coords)
  
def extract_extend_train_patches():
  datadir = './dataset/train'
  infodir = './extented_label'
  files = os.listdir(infodir)
  image = io.imread('./image.tif')
  count = 0
  for file in files:
    classname = os.path.splitext(file)[0]
    label = class2idx[classname]
    path = os.path.join(infodir,file)
    coords = np.loadtxt(path,dtype=np.uint16,skiprows=1)
    for coord in  tqdm(coords):
      x1,y1,x2,y2 = coord
      idxs = np.arange(x1,x2,7,dtype=np.uint16)
      idys = np.arange(y1,y2,7,dtype=np.uint16)
      for idx in idxs:
        for idy in idys:
          count += 1
          patch = image[:,idy:idy+7,idx:idx+7]
          assert patch.shape==(8,7,7)
          np.savez(os.path.join(datadir,'extend_%d.npz'%(count+1)),image=patch,label=label)
          
      
      
  


def extract_margin_test_patches():     
  # simple save margin pixels
  datadir2 = './dataset/test/margin'
  if not os.path.exists(datadir2):
    os.makedirs(datadir2)
    
  image = io.imread('./image.tif')
  
  col_margin = image[:,:-2,-7:]
  assert col_margin.shape==(8,17808,7)
  row_margin = image[:,-7:,:-4]
  assert row_margin.shape==(8,7,50358)
  
  row_max = 2544
  col_max = 7194
  
  columns = np.zeros(shape=[row_max,8,7,7],dtype=np.int16)
  columns_coords = np.zeros(shape=[row_max,2],dtype=np.int16)
  rows = np.zeros(shape=[col_max,8,7,7],dtype=np.int16)
  rows_coords = np.zeros(shape=[col_max,2],dtype=np.int16)
  
  for idx,row in enumerate(range(row_max)):
    x1,x2 = 7*row,7*(row+1)
    patch = col_margin[:,x1:x2,:]
    columns[idx] = patch
    columns_coords[idx] = np.array([row,-1],dtype=np.int16)
  # np.savez(os.path.join(datadir2,'cols.npz'),data=columns,coord=columns_coords)
    
  for idx,col in enumerate(range(col_max)):
    x1,x2 = 7*col,7*(col+1)
    patch = row_margin[...,x1:x2]
    rows[idx] = patch
    rows_coords[idx] = np.array([-1,col],dtype=np.int16)
  
  margins = np.vstack((columns,rows))
  margins_coord = np.vstack((columns_coords,rows_coords))
  
  np.savez(os.path.join(datadir2,'margin.npz'),data=margins,coord=margins_coord)
  
if __name__ == '__main__': 
  # load_info()
  extract_train_patches()
  # extract_test_patches()
  # extract_margin_test_patches()
