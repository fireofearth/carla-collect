import pytest

from collect.in_simulation.midlevel.v8 import MidlevelAgent
from collect.generate.scene.v3_2.trajectron_scene import (
    TrajectronPlusPlusSceneBuilder
)
from tests.Hz20 import PlannerScenario
from tests.Hz20.params import (
    VARIABLES_ph6_step1_ncoin1_np100,
    VARIABLES_ph8_step1_ncoin1_np100,
    VARIABLES_ph6_step1_ncoin1_r_np100,
    VARIABLES_ph6_step1_ncoin1_r_np1000,
    VARIABLES_ph6_step1_ncoin1_r_np5000,
    VARIABLES_ph8_step1_ncoin1_r_np100,
    SCENARIO_scene3_ov1_shift25,
    SCENARIO_scene3_ov1_shift20,
    SCENARIO_scene3_ov1_shift15,
    SCENARIO_scene3_ov1_shift10,
    SCENARIO_scene3_ov4_gap28,
    SCENARIO_scene3_ov4_gap34,
    SCENARIO_scene4_ov1_brake,
    SCENARIO_scene4_ov1_accel,
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
        SCENARIO_scene3_ov4_gap34,
        SCENARIO_scene4_ov1_brake,
        SCENARIO_scene4_ov1_accel,
    ]
)
def test_Town03_scenario(scenario_params, ctrl_params,
    carla_Town03_synchronous, eval_env, eval_stg_cuda
):
    PlannerScenario(
        scenario_params,
        ctrl_params,
        carla_Town03_synchronous,
        eval_env,
        eval_stg_cuda,
        MidlevelAgent,
        TrajectronPlusPlusSceneBuilder
    ).run()
