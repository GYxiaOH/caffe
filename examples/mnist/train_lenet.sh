#!/usr/bin/env sh
set -e

/home/camalab/caffe/build/tools/caffe train --solver=/home/camalab/caffe/examples/mnist/lenet_solver.prototxt $@
