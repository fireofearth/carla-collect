#!/bin/bash

# APPROOT is the path of the carla-collect/ repository root directory
export APPROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set CARLA Simulatory directory manually
export CARLANAME=
export CARLANAME=carla-0.9.11
# export CARLANAME=carla-0.9.13
if [[ -z "$CARLANAME" ]]; then
    echo "Did not set CARLANAME!"
    exit 1
else
    echo "Using CARLA $CARLANAME"
fi
# Directory of CARLA binaries
export CARLA_DIR=/home/$(whoami)/src/$CARLANAME
# CPLEX is only used for in-simulation code
export CPLEX_STUDIO_DIR1210=/opt/ibm/ILOG/CPLEX_Studio1210

# Enable the Python environment
# source $APPROOT/py38trajectron/bin/activate
source $APPROOT/py38torch104trajectron/bin/activate

# Automatic path linking
if [[ "$CARLANAME" == carla-0.9.11 ]]; then
    export PYCARLA=$CARLA_DIR/PythonAPI/carla/dist/$CARLANAME-py3.7-linux-x86_64.egg
fi
export TRAJECTRONPP_DIR=$APPROOT/Trajectron-plus-plus
export UTILITY=$APPROOT/python-utility/utility
export CARLAUTIL=$APPROOT/python-utility/carlautil

# Setting Python path
if [[ "$CARLANAME" == carla-0.9.11 ]]; then
    export PYTHONPATH=$PYCARLA:$UTILITY:$CARLAUTIL:$APPROOT:$PYTHONPATH
else
    export PYTHONPATH=$UTILITY:$CARLAUTIL:$APPROOT:$PYTHONPATH
fi
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes/devkit/python-sdk:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/trajectron:$PYTHONPATH
export PYTHONPATH=$TRAJECTRONPP_DIR/experiments/nuScenes:$PYTHONPATH
# export PYTHONPATH=$CARLA_DIR/PythonAPI/carla:$PYTHONPATH

mkdir -p $APPROOT/out
mkdir -p $APPROOT/cache
