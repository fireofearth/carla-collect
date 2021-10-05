#!/bin/bash

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller.py::test_Town03_scenario[intersection_1-OAAgent_ph8_ch8_np100]

pytest \
	--log-cli-level=INFO \
	--capture=tee-sys \
	tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn-ph8_ch8_np100]


