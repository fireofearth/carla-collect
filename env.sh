#!/bin/bash

export DIMROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CARLANAME=carla-0.9.9
CARLANAME=carla-0.9.11
export CARLA_DIR=/home/$(whoami)/src/$CARLANAME
export PYCARLA=$CARLA_DIR/PythonAPI/carla/dist/$CARLANAME-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYCARLA:$DIMROOT:$PYTHONPATH

mkdir -p $DIMROOT/out
