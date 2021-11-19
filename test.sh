#!/bin/bash

##################
## test controller
##################

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller.py::test_Town03_scenario[intersection_1-OAAgent_ph8_ch8_np100]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller.py::test_Town03_scenario[intersection_1-OAAgent_ph6_ch2_np100]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller.py::test_Town03_scenario[intersection_2-MCCAgent_ph8_ch1_np100_ncoin2]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn-ph8_ch8_np100]

#####################
## test controller v5
#####################

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller_v5.py::test_Town03_scenario[intersection_3-OAAgent_ph6_ch3_np100]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller_v5.py::test_Town03_scenario[intersection_4-OAAgent_ph6_ch1_np100]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller_v5.py::test_Town03_scenario[intersection_5-OAAgent_ph6_ch6_np100]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller_v5.py::test_Town03_scenario[roundabout_1-OAAgent_ph6_ch1_np100]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller_v5.py::test_Town03_scenario[intersection_3-MCCAgent_ph6_ch1_ncoin2_np100]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller_v5.py::test_Town03_scenario[intersection_3-RMCCAgent_ph6_ch1_ncoin2_np100]

#####################
## test controller v6
#####################

pytest \
	--log-cli-level=INFO \
	--capture=tee-sys \
	tests/test_controller_v6.py::test_Town03_scenario[intersection_3-OAAgent_ph6_ch6_np200]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller_v6.py::test_Town03_scenario[intersection_3_1-OAAgent_ph6_ch6_np200]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller_v6.py::test_Town03_scenario[intersection_3_1-OAAgent_ph6_ch6_np1000]

# pytest \
# 	--log-cli-level=INFO \
# 	--capture=tee-sys \
# 	tests/test_controller_v6.py::test_Town03_scenario[intersection_3_1-OAAgent_ph6_ch6_np2000]

##################
## test standalone
##################

# pytest \
#     --log-cli-level=INFO \
#     --capture=tee-sys \
#     tests/test_standalone.py::test_Town03_scenario[intersection_3-ch8_open]

# pytest \
#     --log-cli-level=INFO \
#     --capture=tee-sys \
#     tests/test_standalone.py::test_Town03_scenario[intersection_4_1-ch8_open]

# pytest \
#     --log-cli-level=INFO \
#     --capture=tee-sys \
#     tests/test_standalone.py::test_Town03_scenario[intersection_3-ch8_step1]

# pytest \
#     --log-cli-level=INFO \
#     --capture=tee-sys \
#     tests/test_standalone.py::test_Town03_scenario[intersection_4-ch8_step1]

# pytest \
#     --log-cli-level=INFO \
#     --capture=tee-sys \
#     tests/test_standalone.py::test_Town03_scenario[intersection_5-ch8_step1]

# pytest \
#     --log-cli-level=INFO \
#     --capture=tee-sys \
#     tests/test_standalone.py::test_Town03_scenario[roundabout_1-ch6_step1]


