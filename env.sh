#!/bin/bash

export DIMROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PRECOG_DIR=/home/$(whoami)/code/robotics/precog
export CARLA_DIR=/home/$(whoami)/src/carla
export PYCARLA=$CARLA_DIR/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYCARLA:$DIMROOT:$PRECOG_DIR:$PYTHONPATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DATA_DIR=/media/external/data/precog_generated_dataset/town04/unsorted

mkdir -p $DIMROOT/out