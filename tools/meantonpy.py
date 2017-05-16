#!/usr/bin/env python
# encoding: utf-8

'''

@author: GYxiaOH

@file: pretictedend.py

@time: 17-5-6 下午9:19

@desc:

'''

import sys
caffe_root = '/home/camalab/caffe/'
sys.path.insert(0, '/home/camalab/caffe-ssd/python')
import caffe
import numpy as np

MEAN_PROTO_PATH = '/home/camalab/caffe/models/vgg19/imagenet_mean.binaryproto'               # 待转换的pb格式图像均值文件路径
MEAN_NPY_PATH = '/home/camalab/caffe/models/vgg19/dog_mean.npy'                         # 转换后的numpy格式图像均值文件路径

blob = caffe.proto.caffe_pb2.BlobProto()           # 创建protobuf blob
data = open(MEAN_PROTO_PATH, 'rb' ).read()         # 读入mean.binaryproto文件内容
blob.ParseFromString(data)                         # 解析文件内容到blob

array = np.array(caffe.io.blobproto_to_array(blob))# 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）
mean_npy = array[0]                                # 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
np.save(MEAN_NPY_PATH ,mean_npy)
