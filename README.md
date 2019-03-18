# -Crop-classification
-----------------------------------------
# 目录
-  [1. 赛题要求](#1-赛题要求)
-  [2. 方法与模型](#2-方法与模型)
    -   [方法1](#方法1)
    -   [方法2](#方法2)
    -   [方法3](#方法3)
    -   [方法4](#方法4)

# 1. 赛题要求
 ------------------------------------------
一句话介绍：百度第二届高分杯美丽乡村大赛。  在有少量样本的情况下，对含有三种农作物和背景的一幅多光谱图像进行农作物分类

* 题目介绍

 赛事数据来源于某一时刻一张遥感卫星多光谱图像，覆盖850公里*300公里。要求参赛队利用深度学习等智能算法自动识别出所给图像对应的农作物，包括玉米、大豆、水稻三种农作物区块和其他区块共四种区块，根据参赛团队对场景的识别准确度和时效性进行评分。

* 数据简介

 赛事将提供遥感卫星图像（格式为tif）； 

* 数据类别

 赛事数据来源于某一时刻一张遥感卫星的历史存档数据，是多光谱图像，覆盖850公里*300公里。

* 数据内容

 赛事数据包括：数据内容覆盖玉米、大豆、水稻和其它区块等4类典型农作物场景；玉米、大豆和水稻3类农作物分布在遥感卫星图像中不同区块，不同农作物区块数量不一，不同区块面积大小不一；除以上三类的农作物以外的图像区域定义为其它区块。

 数据格式为多光谱tif图像。赛事目标是从测试数据集中识别出玉米、大豆和水稻所在区块，并把对应的像素点值标注为对应类别的农作物类别值。

* 数据组成

 本次竞赛的数据由原始多光谱图像和训练数据集两部分组成：
 
  原始多光谱图像：

  多光谱图像一张（tif格式），8通道，覆盖面积850公里*300公里

  训练数据集：

  训练数据集是原始多光谱图像中农作物区块的部分标注数据。

  标注样本点数据，给定玉米、大豆和水稻3个类别农作物对应区块的中心点像素位置（x,y）列表，以及对应中心点对应的区块半径3

'''

    样例：FID,Id,作物,半径,备注,x,y

          0,1,玉米,3, ,12500.7001953,-3286.5600586

          1865,1866,大豆,3, ,5941.6601563,-6966.2797852

          2086,2087,水稻,3, ,9165.4697266,-14989.2998047
'''


* 备注：

  1、（x，y）为tif数据格式的坐标系；（x,y）取值为小数，选手可以四舍五入取整获得对应的像素点位置；选手可以从卫星图片中取出对应图片7*7*8作为训练样本；某些农作物区块对应的面积半径可能大于3，选手可以用算法扩展农作物区块面积，作为训练样本；

  2、多光谱图像由多张高分图像拼接而成，会有光照等影响因素使得同类农作物区块的颜色可能不同，此处考验的是选手所训练模型的泛化能力。

  数据对应关系：RasterXSize对应的是x坐标，RasterYSize对应的是y坐标，x坐标步长是1.0，y坐标步长是-1.0，左上角坐标是(0.0, 1e-07)
  
# 2. 方法与模型
 -------------------------------------------
  - ### 方法1
    采用无监督的学习方法，利用自编码器学习样本特征，结合余弦相似度求解
  - ### 方法2
    采用无监督与有监督结合的方法，将自编码器学习到的特征，结合SVM进行分类
  - ### 方法3
    采用自训练的方法[self-training](#self-training)，扩展样本集，训练分类器，进行分类
  - ### 方法4
    采用协同训练的方法[co-training](#co-training)，扩展样本集，训练分类器，进行分类
    
  ### self-training
 
  ### co-training
 
