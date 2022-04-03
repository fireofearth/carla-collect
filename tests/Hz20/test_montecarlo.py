import pytest

from collect.in_simulation.midlevel.v8 import MidlevelAgent
from collect.generate.scene.v3_2.trajectron_scene import (
    TrajectronPlusPlusSceneBuilder
)
from tests.Hz20 import MonteCarloScenario
from tests.Hz20.params import (
    VARIABLES_ph6_step1_ncoin1_np100,
    VARIABLES_ph8_step1_ncoin1_np100,
    VARIABLES_ph6_step1_ncoin1_r_np100,
    VARIABLES_ph6_step1_ncoin1_r_np1000,
    VARIABLES_ph6_step1_ncoin1_r_np5000,
    VARIABLES_ph8_step1_ncoin1_r_np100,
    MONTECARLO_scene4_ov1,
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
        MONTECARLO_scene4_ov1,
    ]
)
def test_Town03_scenario(scenario_params, ctrl_params,
    carla_Town03_synchronous, eval_env, eval_stg_cuda
):
    MonteCarloScenario(
        scenario_params,
        ctrl_params,
        carla_Town03_synchronous,
        eval_env,
        eval_stg_cuda,
        MidlevelAgent,
        TrajectronPlusPlusSceneBuilder
    ).run()
