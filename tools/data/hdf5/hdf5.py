#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:35:50 2017

@author: gyxiaoh
"""

import h5py, os
import sys
sys.path.insert(0, '/home/camalab/caffe-ssd/python')
import caffe
import numpy as np
import random
flag = 1
c = 0

SIZE = 227 # fixed size to all images
with open( 'filename.txt', 'r' ) as T :
    lines = T.readlines()

with open('wordweight.txt') as F:
    th = F.readlines()

with open('type.txt') as R:
    ty = R.readlines()
    
while flag ==1:
    if len(lines) > 3000:
        index = 3000
    else:
        index = len(lines)
        flag = 0
    str1 = str(flag) +' '+str(index)
    print str1
    if index ==0:
        break
    # If you do not have enough memory split data into
    # multiple batches and generate multiple separate h5 files
    X = np.zeros( (index, 3, SIZE, SIZE), dtype='f4' ) 
    y = np.zeros( (index,1000), dtype='f4' )
    z = np.zeros( (index,1), dtype='f4' )
    #for i,l in enumerate(lines):
    for i in range(0,index):
        sp = lines[i].strip('\n')
        labels =th[i].strip('\r\n').split(' ')
        typelabel=ty[i].strip('\n')
        #labels = [random.random() for ii in range(50000)]
        img = caffe.io.load_image(sp)
        img = caffe.io.resize( img, (SIZE, SIZE, 3) ) # resize to fixed size
        # you may apply other input transformations here...
        # Note that the transformation should take img from size-by-size-by-3 and transpose it to 3-by-size-by-size
        # for example
        transposed_img = img.transpose((2,0,1))[::-1,:,:] # RGB->BGR
        X[i] = transposed_img
        #sp =sp[1].split(' ')
        #for i in range(0,len(sp)):
        	#labels.append(sp[i])
        #y[i]=float(sp[1])
        y[i,:] = np.array(labels).astype(np.float32)
        z[i]=float(typelabel)-float(1) #label begin 0
    
    h5name = '/media/camalab/DATADRIVE1/HDF5/'+'train'+str(c)+'.h5'
    with h5py.File(h5name,'w') as H:
        H.create_dataset( 'data', data=X ) # note the name X given to the dataset!
        H['label']=y # note the name y given to the dataset!
	H['typelabel']=z
    c = c+1
    print h5name
    
    for j in range(0,index):
        del lines[0]
        del th[0]
        del ty[0]

with open('/media/camalab/DATADRIVE1/HDF5/train_h5_list.txt','w') as L:
    for line in range(0,c):
        h5filename = '/media/camalab/DATADRIVE1/HDF5/'+'train'+str(line)+'.h5'
        L.write( h5filename ) # list all h5 files you are going to use
	L.write('\n')

