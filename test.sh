#!/bin/bash

pytest tests/Hz20/test_clusters.py
pytest tests/Hz20/test_clusters.py::test_planner_v8.py
pytest tests/Hz20/test_clusters.py::test_planner_v9.py

pytest tests/Hz20/test_planner_v8.py::test_Town03_scenario[scene3_ov4_gap34-ph6_step1_ncoin1_r_np1000]
