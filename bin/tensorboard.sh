#!/bin/bash

# Note: to use 'greadlink', you need to have coreutils installed on your MacOS ("brew install coreutils")
#       if you are on Linux, use 'readlink' instead

SCRIPT=`greadlink -f "$0"`
#SCRIPT=`readlink -f "$0"`
SCRIPT_DIR=`dirname "$SCRIPT"`
APP_HOME="$SCRIPT_DIR"/..

PYTHON_DIR=$APP_HOME/python
BIN_DIR=$APP_HOME/bin

#https://stackoverflow.com/questions/60835400/permission-denied-tmp-tensorboard-info-pid-31318-info-when-trying-to-access
export TMPDIR=/tmp/$USER
mkdir -p $TMPDIR

LOGDIR=$APP_HOME/tb_log
HOST=0.0.0.0
PORT=5050

#https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
export TF_CPP_MIN_LOG_LEVEL=3

#https://github.com/tensorflow/tensorflow/issues/9512#issuecomment-671906467
tensorboard --logdir $LOGDIR --host $HOST --port $PORT --purge_orphaned_data true
