#!/usr/bin/env python
# encoding: utf-8

'''

@author: GYxiaOH

@file: pretictedend.py

@time: 17-5-6 下午9:19

@desc:

'''

import numpy as np

import sys,os
caffe_root = '/home/camalab/caffe/'
sys.path.insert(0, '/home/camalab/caffe-ssd/python')
import caffe
os.chdir(caffe_root)

net_file='/home/camalab/caffe/models/vgg19/deploy_val_click_median.prototxt'
caffe_model='/home/camalab/caffe/models/vgg19/model/_iter_160000.caffemodel'
imageindex = open('/home/camalab/namesortbyclick.txt','r').readlines()
# imagename = open('/home/camalab/filenamesortbyclick.txt','r').readlines()
imagename = open('/home/camalab/filename.txt','r').readlines()
tnumber =open('/home/camalab/numberoftype.txt','r').readlines()
stnumber =open('/home/camalab/stnumber.txt','w')
endfilename=open('/home/camalab/endfilename.txt','w')
gailv=open('/home/camalab/gailv.txt','w')

# for temp in range(0,len(imagename)):
#     imagename[temp] = imagename[temp].strip('\n')
#mean_data=np.array([104,117,123])
mean_npy = np.load('/home/camalab/caffe/models/vgg19/dog_mean.npy')

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data',mean_data)
transformer.set_mean('data',mean_npy.mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead if BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

type = 0
firsthalf = int(int(tnumber[0])/2) - 1
typebeginindex = 0
stemp = 0
for line in range(0,len(imagename)):
    if line > typebeginindex + int(tnumber[type]) - 1:
        typebeginindex = typebeginindex + int(tnumber[type])
        type = type + 1
        firsthalf = typebeginindex + int(int(tnumber[type])/2)
        stnumber.write(str(stemp))
        stnumber.write('\n')
        stemp = 0;
    imgname = imagename[int(imageindex[line])-1].strip('\n')
    image=caffe.io.load_image(imgname)
    image = caffe.io.resize(image, (256, 256, 3))
    image = image[16:239, 16:239, :]
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    out = net.forward()
    output_prob = out['softmax_img1'][0]  #end of predict

    # errorsearch = 0
    # if int(tnumber[type])<51:
    #     if line <  firsthalf:
    #         errorsearch = 1
    #         if output_prob[type] > 0.14:
    #             endfilename.write(imageindex[line])
    #             stemp = stemp + 1
    #             errorsearch = 11
    #            # endfilename.write('\n')
    #     else:
    #         if output_prob[type] > 0.34:
    #             endfilename.write(imageindex[line])
    #             stemp = stemp + 1
    #             errorsearch = 12
    #             #endfilename.write('\n')
    # elif int(tnumber[type])>300:
    #     errorsearch = 2
    #     if line <  firsthalf:
    #         if output_prob[type] > 0.34:
    #             endfilename.write(imageindex[line])
    #             stemp = stemp + 1
    #             errorsearch = 21
    #            # endfilename.write('\n')
    #     else:
    #         if output_prob[type] > 0.54:
    #             endfilename.write(imageindex[line])
    #             stemp = stemp + 1
    #             errorsearch = 22
    #             #endfilename.write('\n')
    # else:
    #     errorsearch = 3
    #     if line < firsthalf:
    #         if output_prob[type] > 0.24:
    #             endfilename.write(imageindex[line])
    #             stemp = stemp + 1
    #             errorsearch = 31
    #             # endfilename.write('\n')
    #     else:
    #         if output_prob[type] > 0.44:
    #             endfilename.write(imageindex[line])
    #             stemp = stemp + 1
    #             errorsearch = 32
    #             # endfilename.write('\n')
    if line < firsthalf:
        if output_prob[type] > 0.29:
            endfilename.write(imageindex[line])
            stemp = stemp + 1
            # endfilename.write('\n')
    else:
        if output_prob[type] > 0.49:
            endfilename.write(imageindex[line])
            stemp = stemp + 1
    print line,type,output_prob[type]
    gailv.write(str(type))
    gailv.write(str(output_prob[type]))
    gailv.write('\n')

stnumber.write(str(stemp))
stnumber.write('\n')
endfilename.close()
gailv.close()
stnumber.close()
