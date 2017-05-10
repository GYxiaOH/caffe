#!/usr/bin/env python
# encoding: utf-8

'''

@author: GYxiaOH

@file: selectpicture.py

@time: 17-5-3 下午1:52

@desc:

'''
import  random

trainf = open('/home/camalab/227HDF5/endfilename.txt', 'r').readlines()
traint = open('/home/camalab/227HDF5/endtype.txt', 'r').readlines()
trainw = open('/home/camalab/227HDF5/wordweight1000.txt', 'r').readlines()

wtrainf = open('/home/camalab/227HDF5/pic4w/filename.txt','w')
wtraint = open('/home/camalab/227HDF5/pic4w/type.txt','w')
wtrainw = open('/home/camalab/227HDF5/pic4w/wordweight.txt','w')

length = len(trainf)

randomindex = range(0, length)

random.shuffle(randomindex)

for i in range(0,len(randomindex)):
    wtrainf.write(trainf[randomindex[i]])
    wtraint.write(traint[randomindex[i]])
    wtrainw.write(trainw[randomindex[i]])


wtrainf.close()
wtraint.close()
wtrainw.close()