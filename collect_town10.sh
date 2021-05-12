#!/bin/bash

SAVEDIR=/home/fireofearth/data/precog_generate/datasets/20210401
    # --augment-data \
python run.py \
    --dir $SAVEDIR \
    --n-vehicles 50 \
    --n-frames 1000 \
    --n-burn-frames 60 \
    --map Town10HD \
