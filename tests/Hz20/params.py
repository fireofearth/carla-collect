import pytest
import carla
import utility as util
import carlautil
from tests import (
    LoopEnum,
    ScenarioParameters,
    CtrlParameters,
)

##################
# Town03 scenarios

STRAIGHT_ROUTES = [
    ["Straight", "Straight", "Straight"],
    ["Straight", "Straight", "Straight"],
    ["Straight", "Straight", "Straight"],
    ["Straight", "Straight", "Straight"],
    ["Straight", "Straight", "Straight"],
]

CONTROLS_scene3 = [
    util.AttrDict(
        interval=(0, 12*10,),
        control=carlautil.create_gear_control(throttle=0.53)
    ),
]

"""Left turn of low curvature to angled road.
OV is shifted 25 m"""
SCENARIO_scene3_ov1_shift25 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[14],
        spawn_shifts=[-5, 25],
        other_routes=STRAIGHT_ROUTES,
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
        other_routes=STRAIGHT_ROUTES,
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
        other_routes=STRAIGHT_ROUTES,
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
        other_routes=STRAIGHT_ROUTES,
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
        other_routes=STRAIGHT_ROUTES,
        n_burn_interval=10,
        run_interval=22,
        controls=CONTROLS_scene3,
        turn_choices=[1],
        max_distance=100,
    ),
    id="scene3_ov4_gap28"
)

"""Left turn of low curvature to angled road with 4 other vehicles.
60 m gap between OVs at the front and OVs at the back."""
SCENARIO_scene3_ov4_gap60 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[15, 14, 15, 14],
        spawn_shifts=[-15, 20, 20, -40, -40],
        other_routes=STRAIGHT_ROUTES,
        n_burn_interval=12,
        run_interval=30,
        controls=CONTROLS_scene3,
        turn_choices=[1],
        max_distance=160,
    ),
    id="scene3_ov4_gap60"
)

CONTROLS_scene4 = [
    util.AttrDict(
        interval=(0, 12*10,),
        control=carla.VehicleControl(throttle=0.6)
    ),
]
SCENARIO_scene4_ov1_brake = pytest.param(
    # Small T-intersection and road bend
    # EV breaks so OV can cross
    ScenarioParameters(
        ego_spawn_idx=89,
        other_spawn_ids=[201],
        other_routes=STRAIGHT_ROUTES,
        spawn_shifts=[-17, 0],
        n_burn_interval=12,
        run_interval=26,
        controls=CONTROLS_scene4,
        turn_choices=[0],
        max_distance=200
    ),
    id="scene4_ov1_brake"
)
SCENARIO_scene4_ov1_fail = pytest.param(
    # Small T-intersection and road bend
    # EV infeasible
    ScenarioParameters(
        ego_spawn_idx=89,
        other_spawn_ids=[201],
        other_routes=STRAIGHT_ROUTES,
        spawn_shifts=[-17, -12],
        n_burn_interval=12,
        run_interval=26,
        controls=CONTROLS_scene4,
        turn_choices=[0],
        max_distance=200
    ),
    id="scene4_ov1_fail"
)
SCENARIO_scene4_ov1_accel = pytest.param(
    # Small T-intersection and road bend
    # EV accelerates to overtake OV.
    # EV can cross road in approx 4 seconds.
    ScenarioParameters(
        ego_spawn_idx=89,
        other_spawn_ids=[201],
        other_routes=STRAIGHT_ROUTES,
        spawn_shifts=[-17, -19],
        n_burn_interval=12,
        run_interval=26,
        controls=CONTROLS_scene4,
        turn_choices=[0],
        max_distance=200
    ),
    id="scene4_ov1_accel"
)
SCENARIO_scene4_ov2_gap55 = pytest.param(
    # Small T-intersection and road bend
    # EV passes between 2 OVs.
    ScenarioParameters(
        ego_spawn_idx=89,
        other_spawn_ids=[201, 201],
        other_routes=STRAIGHT_ROUTES,
        spawn_shifts=[-10, 3, -52],
        n_burn_interval=12,
        run_interval=20,
        controls=CONTROLS_scene4,
        turn_choices=[0],
        max_distance=150
    ),
    id="scene4_ov2_gap55"
)

#####################################
# Scenario parameters for Monte-Carlo

"""Left turn of low curvature to angled road with 4 other vehicles.
34 m gap between vehicles.
Should succeed within 30 steps (average is about 25 steps).
May not traverse intersection within 30 steps if EV yields to vehicles in the back.
"""
MONTEOCARLO_scene3_ov4_gap60 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        other_spawn_ids=[15, 14, 15, 14],
        spawn_shifts=[-15, 20, 20, [-42, -38], [-42, -38]],
        other_routes=STRAIGHT_ROUTES,
        n_burn_interval=12,
        run_interval=30,
        controls=CONTROLS_scene3,
        goal=util.AttrDict(distance=110),
        turn_choices=[1],
        max_distance=120,
    ),
    id="scene3_ov4_gap60"
)

MONTECARLO_scene4_ov1_accel = pytest.param(
    # Small T-intersection and road bend
    # EV accelerates to overtake OV.
    # EV can cross road in approx 4 seconds.
    ScenarioParameters(
        ego_spawn_idx=89,
        other_spawn_ids=[201],
        other_routes=STRAIGHT_ROUTES,
        spawn_shifts=[-17, [-17, -21]],
        n_burn_interval=12,
        run_interval=30,
        controls=CONTROLS_scene4,
        goal=util.AttrDict(distance=100),
        turn_choices=[0],
        max_distance=200
    ),
    id="scene4_ov1_accel"
)

MONTECARLO_scene4_ov1_brake = pytest.param(
    # Small T-intersection and road bend
    # EV breaks so OV can cross
    ScenarioParameters(
        ego_spawn_idx=89,
        other_spawn_ids=[201],
        other_routes=STRAIGHT_ROUTES,
        spawn_shifts=[-17, [-4, 0]],
        n_burn_interval=12,
        run_interval=50,
        controls=CONTROLS_scene4,
        goal=util.AttrDict(distance=100),
        turn_choices=[0],
        max_distance=200
    ),
    id="scene4_ov1_brake"
)

MONTECARLO_scene4_ov2_gap55 = pytest.param(
    # Small T-intersection and road bend
    # EV passes between 2 OVs.
    ScenarioParameters(
        ego_spawn_idx=89,
        other_spawn_ids=[201, 201],
        other_routes=STRAIGHT_ROUTES,
        spawn_shifts=[-17, 3, [-54, -50]],
        n_burn_interval=12,
        run_interval=40,
        controls=CONTROLS_scene4,
        goal=util.AttrDict(distance=100),
        turn_choices=[0],
        max_distance=200
    ),
    id="scene4_ov2_gap55"
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

VARIABLES_ph8_step1_ncoin1_r_np5000 = pytest.param(
    CtrlParameters(
        prediction_horizon=8,
        control_horizon=8,
        step_horizon=1,
        n_predictions=5000,
        n_coincide=1,
        random_mcc=True,
        loop_type=LoopEnum.CLOSED_LOOP
    ),
    id="ph8_step1_ncoin1_r_np5000"
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
