#!/usr/bin/env python
# encoding: utf-8

'''

@author: GYxiaOH

@file: copyselpic.py

@time: 17-5-12 下午1:59

@desc:

'''

import sys
import os
import shutil

filename = open('/home/camalab/filename.txt','r').readlines()
fileindex = open('/home/camalab/endfilename.txt','r').readlines()

objpath = '/media/camalab/DATADRIVE1/selpic/'


for i in range(0,len(fileindex)):
    temfile = filename[int(fileindex[i])-1].strip('\n').split('/')[-2]
    objdirname = objpath + temfile
    if not os.path.exists(objdirname):
        os.mkdir(objdirname)
    temjpg = temfile = filename[int(fileindex[i])-1].strip('\n').split('/')[-1]
    objfilename = objdirname + '/' + temjpg
    oldname = filename[int(fileindex[i])-1].strip('\n')
    shutil.copy(oldname,objfilename)
