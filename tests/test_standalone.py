
import time
import enum
import functools

import pytest
import numpy as np

import carla
import utility as util
import carlautil

from collect.generate import get_all_vehicle_blueprints
from collect.generate import NaiveMapQuerier
from collect.generate.scene import OnlineConfig
from collect.in_simulation.standalone.v1 import MotionPlanner

"""Just test the motion controller itself without the collision avoidance.

pytest \
    --log-cli-level=INFO \
    --capture=tee-sys \
    tests/test_standalone.py::test_Town03_scenario[intersection_3]
"""

class LoopEnum(enum.Enum):
    OPEN_LOOP = 0
    CLOSED_LOOP = 1

class ScenarioParameters(object):
    """
    Attributes
    ==========
    controls : list of util.AttrDict
        Optional deterministic controls to apply to vehicle. Each control has
        attributes `interval` that specifies which frames to apply control,
        and `control` containing a carla.VehicleControl to apply to vehicle.
    run_interval : int
        Number of steps to run motion planner. Only applicable to closed loop.
    goal : util.AttrDict
        Optional goal destination the motion planned vehicle should go to.
        By default the vehicle moves forwards.
        Not applicable to curved road segmented boundary constraints.
    turn_choices : list of int
    max_distance : number
    """

    def __init__(self,
            ego_spawn_idx=None,
            # The distance from original spawn point to place ego vehicle.
            # Set `None` to disable shifting.
            spawn_shift=None,
            # number of timesteps before running
            n_burn_interval=None,
            run_interval=None,
            controls=[],
            goal=None,
            turn_choices=[],
            max_distance=100):
        self.ego_spawn_idx = ego_spawn_idx
        self.spawn_shift = spawn_shift
        self.n_burn_interval = n_burn_interval
        self.run_interval = run_interval
        self.controls = controls
        self.goal = goal
        self.turn_choices = turn_choices
        self.max_distance = max_distance

class CtrlParameters(object):
    def __init__(self,
            control_horizon=8,
            step_horizon=1,
            loop_type=LoopEnum.OPEN_LOOP):
        self.control_horizon = control_horizon
        self.step_horizon = step_horizon
        self.loop_type = loop_type

def shift_spawn_point(carla_map, k, spawn_shifts, spawn_point):
    try:
        spawn_shift = spawn_shifts[k]
        spawn_shift < 0
    except (TypeError, IndexError) as e:
        return spawn_point
    return carlautil.move_along_road(carla_map, spawn_point, spawn_shift)

def scenario(scenario_params, agent_constructor, ctrl_params,
            carla_synchronous, eval_env):
    ego_vehicle = None
    agent = None
    client, world, carla_map, traffic_manager = carla_synchronous

    try:
        map_reader = NaiveMapQuerier(
                world, carla_map, debug=True)
        online_config = OnlineConfig(node_type=eval_env.NodeType)

        # Mock vehicles
        spawn_points = carla_map.get_spawn_points()
        blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
        spawn_point = spawn_points[scenario_params.ego_spawn_idx]
        spawn_point = shift_spawn_point(carla_map,
                0, [scenario_params.spawn_shift], spawn_point)
        spawn_point.location += carla.Location(0, 0, 0.5)
        ego_vehicle = world.spawn_actor(blueprint, spawn_point)

        world.tick()
        agent = agent_constructor(
                ego_vehicle,
                map_reader,
                n_burn_interval=scenario_params.n_burn_interval,
                control_horizon=ctrl_params.control_horizon,
                step_horizon=ctrl_params.step_horizon,
                turn_choices=scenario_params.turn_choices,
                max_distance=scenario_params.max_distance,
                log_agent=False,
                log_cplex=False,
                plot_simulation=True,
                plot_boundary=False,
                road_boundary_constraints=True,
                scene_config=online_config)
        # if scenario_params.goal:
        #     agent.set_goal(scenario_params.goal.x, scenario_params.goal.y,
        #             is_relative=scenario_params.goal.is_relative)
        
        """Move the spectator to the ego vehicle.
        The positioning is a little off"""
        state = agent.get_vehicle_state()
        goal = agent.get_goal()
        goal_x, goal_y = goal.x, -goal.y
        if goal.is_relative:
            location = carla.Location(
                    x=state[0] + goal_x /2.,
                    y=state[1] - goal_y /2.,
                    z=state[2] + 50)
        else:
            location = carla.Location(
                    x=(state[0] + goal_x) /2.,
                    y=(state[1] + goal_y) /2.,
                    z=state[2] + 50)
        world.get_spectator().set_transform(
            carla.Transform(
                location,
                carla.Rotation(pitch=-90)
            )
        )

        n_burn_frames = scenario_params.n_burn_interval*online_config.record_interval
        if ctrl_params.loop_type == LoopEnum.CLOSED_LOOP:
            run_frames = scenario_params.run_interval*online_config.record_interval + 1
        else:
            run_frames = ctrl_params.control_horizon*online_config.record_interval + 1
        for idx in range(n_burn_frames + run_frames):
            control = None
            for ctrl in scenario_params.controls:
                if ctrl.interval[0] <= idx and idx <= ctrl.interval[1]:
                    control = ctrl.control
                    break
            frame = world.tick()
            agent.run_step(frame, control=control)
 
    finally:
        if agent:
            agent.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
    
    time.sleep(1)

##################
# Town03 scenarios

CONTROLS_intersection_4_1 = [
    util.AttrDict(
        interval=(0, 21*5,),
        control=carla.VehicleControl(throttle=0.35)
    ),
    util.AttrDict(
        interval=(22, 25*5,),
        control=carla.VehicleControl(throttle=0.3, steer=-0.2)
    ),
]

SCENARIO_intersection_4_1 = pytest.param(
    # left turn of low curvature to angled road
    ScenarioParameters(
            ego_spawn_idx=85,
            n_burn_interval=26,
            run_interval=13,
            controls=CONTROLS_intersection_4_1,
            turn_choices=[2],
            max_distance=70),
    id="intersection_4_1"
)

CONTROLS_intersection_3 = [
    util.AttrDict(
        interval=(0, 11*5,),
        control=carla.VehicleControl(throttle=0.3)
    ),
]
SCENARIO_intersection_3 = pytest.param(
    # left turn of low curvature to angled road
    ScenarioParameters(
            ego_spawn_idx=85,
            n_burn_interval=12,
            run_interval=13,
            controls=CONTROLS_intersection_3,
            turn_choices=[1],
            max_distance=70),
    id="intersection_3"
)
SCENARIO_intersection_4 = pytest.param(
    # simple left turn
    ScenarioParameters(
            ego_spawn_idx=85,
            n_burn_interval=12,
            run_interval=15,
            controls=CONTROLS_intersection_3,
            turn_choices=[2],
            max_distance=70),
    id="intersection_4"
)
SCENARIO_intersection_5 = pytest.param(
    # simple right turn
    ScenarioParameters(
            ego_spawn_idx=85,
            n_burn_interval=12,
            run_interval=24,
            controls=CONTROLS_intersection_3,
            turn_choices=[3],
            max_distance=70),
    id="intersection_5"
)

CONTROLS_roundabout_1 = [
    util.AttrDict(
        interval=(0, 9*5,),
        control=carla.VehicleControl(throttle=0.48)
    ),
]
SCENARIO_roundabout_1 = pytest.param(
    ScenarioParameters(
            ego_spawn_idx=247,
            n_burn_interval=5,
            run_interval=40,
            controls=CONTROLS_roundabout_1,
            turn_choices=[0, 1],
            max_distance=200),
    id="roundabout_1"
)

VARIABLES_ch8_step1 = pytest.param(
    MotionPlanner, CtrlParameters(
        control_horizon=8,
        step_horizon=1,
        loop_type=LoopEnum.CLOSED_LOOP
    ),
    id="ch8_step1"
)
VARIABLES_ch6_step1 = pytest.param(
    MotionPlanner, CtrlParameters(
        control_horizon=6,
        step_horizon=1,
        loop_type=LoopEnum.CLOSED_LOOP
    ),
    id="ch6_step1"
)
VARIABLES_ch8_open = pytest.param(
    MotionPlanner, CtrlParameters(
        control_horizon=8,
        step_horizon=8,
        loop_type=LoopEnum.OPEN_LOOP
    ),
    id="ch8_open"
)
VARIABLES_ch6_open = pytest.param(
    MotionPlanner, CtrlParameters(
        control_horizon=6,
        step_horizon=6,
        loop_type=LoopEnum.OPEN_LOOP
    ),
    id="ch6_open"
)

@pytest.mark.parametrize(
    "agent_constructor,ctrl_params",
    [
        VARIABLES_ch8_step1,
        VARIABLES_ch6_step1,
        VARIABLES_ch8_open,
        VARIABLES_ch6_open
    ]
)
@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_intersection_3,
        SCENARIO_intersection_4,
        SCENARIO_intersection_5,
        SCENARIO_intersection_4_1,
        SCENARIO_roundabout_1
    ]
)
def test_Town03_scenario(scenario_params, agent_constructor, ctrl_params,
        carla_Town03_synchronous, eval_env):
    scenario(scenario_params, agent_constructor, ctrl_params,
            carla_Town03_synchronous, eval_env)
