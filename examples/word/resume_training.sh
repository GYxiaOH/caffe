#!/usr/bin/env sh

LOG=/media/camalab/DATADRIVE1/wordmodel/log/-`date +%Y-%m-%d-%H-%M-%S`.log

/home/camalab/caffe/build/tools/caffe train  --solver=/home/camalab/caffe/examples/word/solver.prototxt  2>&1 | tee $LOG
