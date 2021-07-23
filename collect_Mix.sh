#!/bin/bash

conda activate trajectron-cplex
source env.sh

# python synthesize.py

sleep 3

python synthesize.py \
        --start-at-episode 5 \
        --label Town04 \
        --map Town04 \
        --n-vehicles 220 \
        --n-burn-frames 500 \
        --n-frames 1000

sleep 3

python synthesize.py \
        --start-at-episode 10 \
        --label Town05 \
        --map Town05 \
        --n-vehicles 180 \
        --n-burn-frames 500 \
        --n-frames 1000

sleep 3

python synthesize.py \
        --start-at-episode 15 \
        --label Town06 \
        --map Town06 \
        --n-vehicles 180

sleep 3

python synthesize.py \
        --start-at-episode 20 \
        --label Town06 \
        --map Town06 \
        --n-vehicles 180

sleep 3

python synthesize.py \
        --start-at-episode 25 \
        --label Town07 \
        --map Town07 \
        --n-vehicles 50

sleep 3

python synthesize.py \
        --start-at-episode 30 \
        --label Town10HD \
        --map Town10HD \
        --n-vehicles 50