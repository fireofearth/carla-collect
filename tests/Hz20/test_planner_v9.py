import pytest

import carla
import utility as util

from collect.in_simulation.midlevel.v9 import MidlevelAgent
from collect.generate.scene.v3_2.trajectron_scene import (
    TrajectronPlusPlusSceneBuilder
)
from tests.Hz20 import PlannerScenario
from tests.Hz20.params import (
    VARIABLES_ph6_step1_ncoin1_np100,
    VARIABLES_ph8_step1_ncoin1_np100,
    VARIABLES_ph6_step1_ncoin1_r_np100,
    VARIABLES_ph8_step1_ncoin1_r_np100,
    SCENARIO_intersection_3,
    SCENARIO_intersection_3_1,
    SCENARIO_intersection_3_2,
)

@pytest.mark.parametrize(
    "ctrl_params",
    [
        VARIABLES_ph6_step1_ncoin1_np100,
        VARIABLES_ph8_step1_ncoin1_np100,
        VARIABLES_ph6_step1_ncoin1_r_np100,
        VARIABLES_ph8_step1_ncoin1_r_np100,
    ]
)
@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_intersection_3,
        SCENARIO_intersection_3_1,
        SCENARIO_intersection_3_2,
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
