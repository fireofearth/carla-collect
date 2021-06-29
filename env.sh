#!/bin/bash

export APPROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CARLANAME=carla-0.9.9
CARLANAME=carla-0.9.11
export CARLA_DIR=/home/$(whoami)/src/$CARLANAME
export PYCARLA=$CARLA_DIR/PythonAPI/carla/dist/$CARLANAME-py3.7-linux-x86_64.egg

export TRAJECTRONPP_DIR=/home/$(whoami)/code/robotics/trajectron-plus-plus

export CPLEX_STUDIO_DIR1210=/opt/ibm/ILOG/CPLEX_Studio1210
export PYTHONPATH=$PYCARLA:$APPROOT:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes/devkit/python-sdk:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/trajectron:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes:$PYTHONPATH
export PYTHONPATH=$CARLA_DIR/PythonAPI/carla:$PYTHONPATH

mkdir -p $APPROOT/out
