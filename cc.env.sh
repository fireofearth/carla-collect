#!/bin/bash

export APPROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Must set this manually
CARLANAME=carla-0.9.11
export CARLA_DIR=/home/cchen795/scratch/src/$CARLANAME/$CARLANAME

# Automatic path linking
export PYCARLA=$CARLA_DIR/PythonAPI/carla/dist/$CARLANAME-py3.7-linux-x86_64.egg
export TRAJECTRONPP_DIR=$APPROOT/Trajectron-plus-plus
export UTILITY=$APPROOT/python-utility/utility
export CARLAUTIL=$APPROOT/python-utility/carlautil

# Setting Python path
export PYTHONPATH=$UTILITY:$CARLAUTIL:$PYCARLA:$APPROOT:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes/devkit/python-sdk:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/trajectron:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes:$PYTHONPATH
export PYTHONPATH=$CARLA_DIR/PythonAPI/carla:$PYTHONPATH

mkdir -p $APPROOT/out
