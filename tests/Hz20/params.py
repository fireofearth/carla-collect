import pytest
import carla
import utility as util
from tests import (
    LoopEnum,
    ScenarioParameters,
    CtrlParameters,
)

##################
# Town03 scenarios

CONTROLS_scene3 = [
    util.AttrDict(
        interval=(0, 9*10,),
        control=carla.VehicleControl(throttle=0.4)
    ),
]

"""Left turn of low curvature to angled road.
OV is shifted 25 m"""
SCENARIO_scene3_ov1_shift25 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14],
        spawn_shifts=[-5, 25],
        n_burn_interval=10,
        run_interval=8,
        controls=CONTROLS_scene3,
        turn_choices=[1],
        max_distance=75,
    ),
    id="scene3_ov1_shift25"
)

"""Left turn of low curvature to angled road.
OV is shifted 20 m"""
SCENARIO_scene3_ov1_shift20 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14],
        spawn_shifts=[-5, 20],
        n_burn_interval=10,
        run_interval=8,
        controls=CONTROLS_scene3,
        turn_choices=[1],
        max_distance=75,
    ),
    id="scene3_ov1_shift20"
)

"""Left turn of low curvature to angled road.
OV is shifted 15 m"""
SCENARIO_scene3_ov1_shift15 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14],
        spawn_shifts=[-5, 15],
        n_burn_interval=10,
        run_interval=8,
        controls=CONTROLS_scene3,
        turn_choices=[1],
        max_distance=75,
    ),
    id="scene3_ov1_shift15"
)

"""Left turn of low curvature to angled road.
OV is shifted 10 m"""
SCENARIO_scene3_ov1_shift10 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14],
        spawn_shifts=[-5, 10],
        n_burn_interval=14,
        run_interval=10,
        controls=CONTROLS_scene3,
        turn_choices=[1],
        max_distance=75,
    ),
    id="scene3_ov1_shift10"
)

"""Left turn of low curvature to angled road with 4 other vehicles.
28 m gap between vehicles.

Notes
=====
Causes MCC simulation to crash when accounting
for permutations of contingency trajectories."""
SCENARIO_scene3_ov4_gap28 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14, 14, 15, 15],
        spawn_shifts=[-5, 31, 23, -5, -13],
        n_burn_interval=10,
        run_interval=22,
        controls=CONTROLS_scene3,
        turn_choices=[1],
        max_distance=100,
    ),
    id="scene3_ov4_gap28"
)

"""Left turn of low curvature to angled road with 4 other vehicles.
34 m gap between vehicles."""
SCENARIO_scene3_ov4_gap34 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14, 14, 15, 15],
        spawn_shifts=[-5, 31, 23, -11, -19],
        n_burn_interval=10,
        run_interval=22,
        controls=CONTROLS_scene3,
        turn_choices=[1],
        max_distance=100,
    ),
    id="scene3_ov4_gap34"
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

VARIABLES_ph6_step1_ncoin1_r_np1000 = pytest.param(
    CtrlParameters(
        prediction_horizon=6,
        control_horizon=6,
        step_horizon=1,
        n_predictions=1000,
        n_coincide=1,
        random_mcc=True,
        loop_type=LoopEnum.CLOSED_LOOP
    ),
    id="ph6_step1_ncoin1_r_np1000"
)

VARIABLES_ph6_step1_ncoin1_r_np5000 = pytest.param(
    CtrlParameters(
        prediction_horizon=6,
        control_horizon=6,
        step_horizon=1,
        n_predictions=5000,
        n_coincide=1,
        random_mcc=True,
        loop_type=LoopEnum.CLOSED_LOOP
    ),
    id="ph6_step1_ncoin1_r_np5000"
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
