#!/bin/bash

test_few_other_vehicles () {
    python run.py -v \
        --n-vehicles 2 \
        --n-data-collectors 1 \
        --n-episodes 1
}

test_few_other_vehicles