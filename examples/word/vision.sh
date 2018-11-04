#!/usr/bin/env sh

LOG=/media/ghat/新加卷/wordmodel/log/vision/-`date +%Y-%m-%d-%H-%M-%S`.log

/home/ghat/caffe/build/tools/caffe train --weights=/home/ghat/caffe/examples/word/visionnotfinetune.caffemodel --solver=/home/ghat/caffe/examples/word/vision.prototxt 2>&1 | tee $LOG 
