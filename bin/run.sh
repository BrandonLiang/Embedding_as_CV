#!/bin/bash

# Note: to use 'greadlink', you need to have coreutils installed on your MacOS ("brew install coreutils")
#       if you are on Linux, use 'readlink' instead

SCRIPT=`greadlink -f "$0"`
#SCRIPT=`readlink -f "$0"`
SCRIPT_DIR=`dirname "$SCRIPT"`
APP_HOME="$SCRIPT_DIR"/..

PYTHON_DIR=$APP_HOME/python
CONFIG_DIR=$APP_HOME/config
BIN_DIR=$APP_HOME/bin

python $PYTHON_DIR/run.py \
  -c $CONFIG_DIR/config.json
  -r $APP_HOME/checkpoint
