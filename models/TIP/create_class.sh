#!/usr/bin/env sh

LOG=/home/hongyuan/l2/-`date +%Y-%m-%d-%H-%M-%S`.log

/home/hongyuan/zhy/caffe/build/tools/caffe train --weights=/home/hongyuan/zhy/caffe/models/vggs/VGG_CNN_S_2.caffemodel --solver=./solver_click_median.prototxt --gpu 2 2>&1 | tee $LOG 