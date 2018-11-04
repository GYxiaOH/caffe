#!/usr/bin/env sh

LOG=/media/ghat/新加卷/wordmodel/log/-`date +%Y-%m-%d-%H-%M-%S`.log

/home/ghat/caffe/build/tools/caffe train --weights=/home/ghat/caffe/examples/word/_iter_300000.caffemodel --solver=/home/ghat/caffe/examples/word/buping.prototxt 2>&1 | tee $LOG 
