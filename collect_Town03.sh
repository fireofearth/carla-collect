#!/bin/bash

conda activate trajectron-cplex
source env.sh

python synthesize.py

sleep 3

python synthesize.py --start-at-episode 5

sleep 3

python synthesize.py --start-at-episode 10

sleep 3

python synthesize.py --start-at-episode 15

sleep 3

python synthesize.py --start-at-episode 20
