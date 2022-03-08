import pytest
import carla
import utility as util
from tests import (
    LoopEnum, ScenarioParameters, CtrlParameters
)

##################
# Town03 scenarios

CONTROLS_intersection_3 = [
    util.AttrDict(
        interval=(0, 9*10,),
        control=carla.VehicleControl(throttle=0.4)
    ),
]

"""Left turn of low curvature to angled road."""
SCENARIO_intersection_3 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14],
        spawn_shifts=[-5, 25],
        n_burn_interval=10,
        run_interval=8,
        controls=CONTROLS_intersection_3,
        turn_choices=[1],
        max_distance=75,
    ),
    id="intersection_3"
)

"""Left turn of low curvature to angled road with 4 other vehicles.
Causes MCC simulation to crash."""
SCENARIO_intersection_3_1 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14, 14, 15, 15],
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
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14, 14, 15, 15],
        spawn_shifts=[-5, 31, 23, -11, -19],
        n_burn_interval=10,
        run_interval=22,
        controls=CONTROLS_intersection_3,
        turn_choices=[1],
        max_distance=100,
    ),
    id="intersection_3_2"
)

####################
# Control parameters

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

VARIABLES_ph8_step1_ncoin1_np100 = pytest.param(
    CtrlParameters(
        prediction_horizon=8,
        control_horizon=8,
        step_horizon=1,
        n_predictions=100,
        n_coincide=1,
        random_mcc=False,
        loop_type=LoopEnum.CLOSED_LOOP
    ),
    id="ph8_step1_ncoin1_np100"
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

VARIABLES_ph8_step1_ncoin1_r_np100 = pytest.param(
    CtrlParameters(
        prediction_horizon=8,
        control_horizon=8,
        step_horizon=1,
        n_predictions=100,
        n_coincide=1,
        random_mcc=True,
        loop_type=LoopEnum.CLOSED_LOOP
    ),
    id="ph8_step1_ncoin1_r_np100"
)
