#!/bin/bash

TOOLS=../../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

LOG_logtostderr=0 GLOG_log_dir=log/  $TOOLS/caffe train -solver lstm_solver_vgg.prototxt -gpu 0
#-snapshot models/_iter_10000.solverstate -gpu 1

echo "Done."
