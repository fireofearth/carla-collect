#!/bin/bash

source lab.env.sh

# python synthesize.py \
#     --label Town03 \
#     --map Town03

# sleep 3

python synthesize.py \
    --start-at-episode 5 \
    --label Town03 \
    --map Town03

sleep 3

python synthesize.py \
    --start-at-episode 10 \
    --label Town03 \
    --map Town03

sleep 3

python synthesize.py \
    --start-at-episode 15 \
    --label Town03 \
    --map Town03

sleep 3

python synthesize.py \
    --start-at-episode 20 \
    --label Town03 \
    --map Town03
