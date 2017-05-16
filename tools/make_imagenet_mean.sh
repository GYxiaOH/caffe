#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/media/camalab/DATADRIVE1/LMDB
DATA=/home/camalab/caffe/models/vgg19
TOOLS=/home/camalab/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
