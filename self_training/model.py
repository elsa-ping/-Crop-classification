#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:19:11 2019

@author: liang
"""

# build model
import torch
from torch import nn
import torch.nn.functional as F

class TinyNet(nn.Module):
  
  def __init__(self):
    super(TinyNet,self).__init__()
    self.features = nn.Sequential(
                                # conv1_block ->5x5x32
                                nn.Conv2d(8,16,3,bias=False),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True),
                                nn.InstanceNorm2d(16,affine=True),                             
                                nn.ReLU(inplace=True),
                                # conv2_block ->3x3x64
                                nn.Conv2d(16,32,3,bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(True),
                                nn.InstanceNorm2d(32,affine=True),
                                nn.ReLU(True),
                                # conv3_block ->1x1x128
                                nn.Conv2d(32,64,3,bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(True)
                              )
    self.classifier = nn.Linear(64,4)
    
  def forward(self,xs):
    bs = xs.size(0)
    xs = self.features(xs)
    xs = xs.view(bs,-1)
    xs = self.classifier(xs)
    return xs
  
if __name__ == '__main__':
  tinynet = TinyNet()
  print(tinynet)
  xs = torch.randn(size=[128,8,7,7])
  out = tinynet(xs)
  print(out.size())