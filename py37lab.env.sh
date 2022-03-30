#!/bin/bash

# APPROOT is the path of the carla-collect/ repository root directory
export APPROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set CARLA Simulatory directory manually
export CARLANAME=carla-0.9.13
export CARLA_DIR=/home/$(whoami)/colinc/$CARLANAME
# CPLEX is optional: it is only used for in-simulation code
export CPLEX_STUDIO_DIR1210=/opt/ibm/ILOG/CPLEX_Studio1210

# Enable the Python environment
source $APPROOT/py37trajectron/bin/activate

# Automatic path linking
# pip install pip
# export PYCARLA=$CARLA_DIR/PythonAPI/carla/dist/$CARLANAME-py3.7-linux-x86_64.egg
export TRAJECTRONPP_DIR=$APPROOT/Trajectron-plus-plus
export UTILITY=$APPROOT/python-utility/utility
export CARLAUTIL=$APPROOT/python-utility/carlautil

# Setting Python path
# export PYTHONPATH=$PYCARLA:$UTILITY:$CARLAUTIL:$APPROOT:$PYTHONPATH
export PYTHONPATH=$UTILITY:$CARLAUTIL:$APPROOT:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes/devkit/python-sdk:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/trajectron:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes:$PYTHONPATH
# export PYTHONPATH=$CARLA_DIR/PythonAPI/carla:$PYTHONPATH

mkdir -p $APPROOT/out
