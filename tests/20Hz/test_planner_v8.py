import pytest

import carla
import utility as util

from tests import (
    LoopEnum, ScenarioParameters, CtrlParameters
)
from . import PlannerScenario
from collect.generate import get_all_vehicle_blueprints
from collect.generate import NaiveMapQuerier
from collect.in_simulation.midlevel.v8 import MidlevelAgent
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v3_2.trajectron_scene import (
    TrajectronPlusPlusSceneBuilder
)

##################
# Town03 scenarios

CONTROLS_intersection_3 = [
    util.AttrDict(
        interval=(0, 9*10,),
        control=carla.VehicleControl(throttle=0.4)
    ),
]
SCENARIO_intersection_3 = pytest.param(
    # left turn of low curvature to angled road
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14],
        spawn_shifts=[-5, 17],
        n_burn_interval=5,
        run_interval=25,
        controls=CONTROLS_intersection_3,
        turn_choices=[1],
        max_distance=75,
    ),
    id="intersection_3"
)
SCENARIO_intersection_3_1 = pytest.param(
    # left turn of low curvature to angled road
    # 4 other vehicles
    # Causes MCC to crash
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14, 14, 15, 15],
        # spawn_shifts=[-5, 31, 23, -11, -19],
        spawn_shifts=[-5, 31, 23, -5, -13],
        n_burn_interval=10,
        run_interval=22,
        controls=CONTROLS_intersection_3,
        turn_choices=[1],
        max_distance=100,
    ),
    id="intersection_3_1"
)
SCENARIO_intersection_3_2 = pytest.param(
    # left turn of low curvature to angled road
    # 4 other vehicles
    # Causes MCC to crash
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14, 14],
        spawn_shifts=[-5, 31-5, 23-5],
        n_burn_interval=10,
        run_interval=22,
        controls=CONTROLS_intersection_3,
        turn_choices=[1],
        max_distance=100,
    ),
    id="intersection_3_2"
)

VARIABLES_ph6_step1_ncoin1_np100 = pytest.param(
    CtrlParameters(
        prediction_horizon=6,
        control_horizon=6,
        step_horizon=1,
        n_predictions=100,
        n_coincide=1,
        random_mcc=False,
        loop_type=LoopEnum.CLOSED_LOOP
    ),
    id="ph6_step1_ncoin1_np100"
)
VARIABLES_ph6_step1_ncoin1_r_np100 = pytest.param(
    CtrlParameters(
        prediction_horizon=6,
        control_horizon=6,
        step_horizon=1,
        n_predictions=100,
        n_coincide=1,
        random_mcc=True,
        loop_type=LoopEnum.CLOSED_LOOP
    ),
    id="ph6_step1_ncoin1_r_np100"
)

@pytest.mark.parametrize(
    "ctrl_params",
    [
        VARIABLES_ph6_step1_ncoin1_np100,
        VARIABLES_ph6_step1_ncoin1_r_np100
    ]
)
@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_intersection_3,
        SCENARIO_intersection_3_1,
        SCENARIO_intersection_3_2
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
