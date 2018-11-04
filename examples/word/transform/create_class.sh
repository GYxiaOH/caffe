#!/usr/bin/env sh

LOG=/media/camalab/DATADRIVE1/wordmodel/log/-`date +%Y-%m-%d-%H-%M-%S`.log

/home/camalab/caffe/build/tools/caffe train --weights=/home/camalab/caffe/examples/word/transform/alexnet_cvgj_iter_320000.caffemodel --solver=/home/camalab/caffe/examples/word/transform/classsolver.prototxt  2>&1 | tee $LOG 
