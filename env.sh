#!/bin/bash

export APPROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Must set these paths manually
CARLANAME=carla-0.9.9
CARLANAME=carla-0.9.11
export CARLA_DIR=/home/$(whoami)/src/$CARLANAME
export CPLEX_STUDIO_DIR1210=/opt/ibm/ILOG/CPLEX_Studio1210

# Automatic path linking
export PYCARLA=$CARLA_DIR/PythonAPI/carla/dist/$CARLANAME-py3.7-linux-x86_64.egg
export TRAJECTRONPP_DIR=$APPROOT/Trajectron-plus-plus

# Setting Python path
export PYTHONPATH=$PYCARLA:$APPROOT:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes/devkit/python-sdk:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/trajectron:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes:$PYTHONPATH
export PYTHONPATH=$CARLA_DIR/PythonAPI/carla:$PYTHONPATH

mkdir -p $APPROOT/out
conda activate trajectron-cplex
