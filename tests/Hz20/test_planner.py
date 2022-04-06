import pytest

from collect.in_simulation.midlevel.v8 import MidlevelAgent as MidlevelAgentV8
from collect.in_simulation.midlevel.v9 import MidlevelAgent as MidlevelAgentV9
from collect.generate.scene.v3_2.trajectron_scene import TrajectronPlusPlusSceneBuilder
from tests.Hz20 import PlannerScenario
from tests.Hz20.params import (
    VARIABLES_ph6_step1_ncoin1_np100,
    VARIABLES_ph8_step1_ncoin1_np100,
    VARIABLES_ph6_step1_ncoin1_r_np100,
    VARIABLES_ph6_step1_ncoin1_r_np1000,
    VARIABLES_ph6_step1_ncoin1_r_np5000,
    VARIABLES_ph8_step1_ncoin1_r_np100,
    VARIABLES_ph8_step1_ncoin1_r_np5000,
    SCENARIO_scene3_ov1_shift25,
    SCENARIO_scene3_ov1_shift20,
    SCENARIO_scene3_ov1_shift15,
    SCENARIO_scene3_ov1_shift10,
    SCENARIO_scene3_ov4_gap28,
    SCENARIO_scene3_ov4_gap60,
    SCENARIO_scene4_ov1_brake,
    SCENARIO_scene4_ov1_accel,
    SCENARIO_scene4_ov2_gap55,
)

MIDLEVEL_v8 = pytest.param(
    MidlevelAgentV8, id="v8"
)
MIDLEVEL_v9 = pytest.param(
    MidlevelAgentV9, id="v9"
)

@pytest.mark.parametrize(
    "ctrl_params",
    [
        VARIABLES_ph6_step1_ncoin1_np100,
        VARIABLES_ph8_step1_ncoin1_np100,
        VARIABLES_ph6_step1_ncoin1_r_np100,
        VARIABLES_ph6_step1_ncoin1_r_np1000,
        VARIABLES_ph6_step1_ncoin1_r_np5000,
        VARIABLES_ph8_step1_ncoin1_r_np100,
        VARIABLES_ph8_step1_ncoin1_r_np5000,
    ]
)
@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_scene3_ov1_shift25,
        SCENARIO_scene3_ov1_shift20,
        SCENARIO_scene3_ov1_shift15,
        SCENARIO_scene3_ov1_shift10,
        SCENARIO_scene3_ov4_gap28,
        SCENARIO_scene3_ov4_gap60,
        SCENARIO_scene4_ov1_brake,
        SCENARIO_scene4_ov1_accel,
        SCENARIO_scene4_ov2_gap55,
    ]
)
@pytest.mark.parametrize(
    "midlevel_agent",
    [MIDLEVEL_v8, MIDLEVEL_v9]
)
def test_Town03_scenario(midlevel_agent, scenario_params, ctrl_params,
    carla_Town03_synchronous, eval_env, eval_stg_cuda
):
    PlannerScenario(
        scenario_params,
        ctrl_params,
        carla_Town03_synchronous,
        eval_env,
        eval_stg_cuda,
        midlevel_agent,
        TrajectronPlusPlusSceneBuilder
    ).run()
