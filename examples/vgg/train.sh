#!/usr/bin/env sh

LOG=/media/camalab/DATADRIVE1/wordmodel/log/-`date +%Y-%m-%d-%H-%M-%S`.log

/home/camalab/caffe/build/tools/caffe train --weights=/home/camalab/caffe/models/vgg19/VGG_ILSVRC_19_layers.caffemodel --solver=/home/camalab/caffe/models/vgg19/solver2.prototxt 2>&1 | tee $LOG 
