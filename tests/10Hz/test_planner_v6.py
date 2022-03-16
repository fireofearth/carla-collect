
import os
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
from collect.in_simulation.midlevel.v6 import MidlevelAgent
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v3_2.trajectron_scene import (
        TrajectronPlusPlusSceneBuilder)

"""Test the midlevel planner v6.

pytest \
    --log-cli-level=INFO \
    --capture=tee-sys \
    tests/test_planner_v6.py::test_Town03_scenario[intersection_3-OAAgent_ph4_ch1_np100]
"""

class OAAgent(MidlevelAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, agent_type="oa", **kwargs)

class MCCAgent(MidlevelAgent):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MCC for v6 is not done.")
        # super().__init__(*args, agent_type="mcc", **kwargs)

class RMCCAgent(MidlevelAgent):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RMCC for v6 is not done.")
        # super().__init__(*args, agent_type="rmcc", **kwargs)

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
    ignore_signs : bool
        Ignore stop signs.
    ignore_lights : bool
        Ignore traffic lights.
    """

    def __init__(self,
            ego_spawn_idx=None,
            other_spawn_ids=[],
            # spawn shifts for the vehicles i=1,2,... in
            # `[ego_spawn_idx] + other_spawn_ids`.
            # Value of `spawn_shifts[i]` is the distance
            # from original spawn point to place vehicle i.
            # Let `spawn_shifts[i] = None` to disable shifting.
            spawn_shifts=[],
            # number of timesteps before running
            n_burn_interval=None,
            run_interval=None,
            controls=[],
            goal=None,
            turn_choices=[],
            max_distance=100,
            ignore_signs=True,
            ignore_lights=True,
            ignore_vehicles=True,
            auto_lane_change=False,
            loop_type=LoopEnum.OPEN_LOOP):
        self.ego_spawn_idx = ego_spawn_idx
        self.other_spawn_ids = other_spawn_ids
        self.spawn_shifts = spawn_shifts
        self.n_burn_interval = n_burn_interval
        self.run_interval = run_interval
        self.controls = controls
        self.goal = goal
        self.turn_choices = turn_choices
        self.max_distance = max_distance
        self.ignore_signs = ignore_signs
        self.ignore_lights = ignore_lights
        self.ignore_vehicles = ignore_vehicles
        self.auto_lane_change = auto_lane_change
        self.loop_type = loop_type

class CtrlParameters(object):
    def __init__(self,
            prediction_horizon=8,
            control_horizon=8,
            step_horizon=1,
            n_predictions=100,
            # not applicable for OAAgent
            n_coincide=4):
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.step_horizon = step_horizon
        self.n_predictions = n_predictions
        self.n_coincide = n_coincide

def shift_spawn_point(carla_map, k, spawn_shifts, spawn_point):
    try:
        spawn_shift = spawn_shifts[k]
        spawn_shift < 0
    except (TypeError, IndexError) as e:
        return spawn_point
    return carlautil.move_along_road(carla_map, spawn_point, spawn_shift)


def attach_camera_to_spectator(world, frame):
    os.makedirs(f"out/starting{frame}", exist_ok=True)
    blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    blueprint.set_attribute('image_size_x', '512')
    blueprint.set_attribute('image_size_y', '512')
    blueprint.set_attribute('fov', '80')
    blueprint.set_attribute('sensor_tick', '0.2')
    sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=world.get_spectator())
    def take_picture(image):
        image.save_to_disk(f"out/starting{frame}/frame{image.frame}_spectator.png")
    sensor.listen(take_picture)
    return sensor

def scenario(scenario_params, agent_constructor, ctrl_params,
            carla_synchronous, eval_env, eval_stg):
    ego_vehicle = None
    agent = None
    spectator_camera = None
    other_vehicles = []
    record_spectator = False
    client, world, carla_map, traffic_manager = carla_synchronous

    try:
        map_reader = NaiveMapQuerier(
                world, carla_map, debug=True)
        online_config = OnlineConfig(node_type=eval_env.NodeType)

        # Mock vehicles
        spawn_points = carla_map.get_spawn_points()
        blueprints = get_all_vehicle_blueprints(world)
        spawn_indices = [scenario_params.ego_spawn_idx] + scenario_params.other_spawn_ids
        other_vehicle_ids = []
        for k, spawn_idx in enumerate(spawn_indices):
            if k == 0:
                blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
            else:
                blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[spawn_idx]
            spawn_point = shift_spawn_point(carla_map,
                    k, scenario_params.spawn_shifts, spawn_point)
            # Prevent collision with road.
            spawn_point.location += carla.Location(0, 0, 0.5)
            vehicle = world.spawn_actor(blueprint, spawn_point)
            if k == 0:
                ego_vehicle = vehicle
            else:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                if scenario_params.ignore_signs:
                    traffic_manager.ignore_signs_percentage(vehicle, 100.)
                if scenario_params.ignore_lights:
                    traffic_manager.ignore_lights_percentage(vehicle, 100.)
                if scenario_params.ignore_vehicles:
                    traffic_manager.ignore_vehicles_percentage(vehicle, 100.)
                if not scenario_params.auto_lane_change:
                    traffic_manager.auto_lane_change(vehicle, False)
                other_vehicles.append(vehicle)
                other_vehicle_ids.append(vehicle.id)

        frame = world.tick()
        agent = agent_constructor(
                ego_vehicle,
                map_reader,
                other_vehicle_ids,
                eval_stg,
                n_burn_interval=scenario_params.n_burn_interval,
                control_horizon=ctrl_params.control_horizon,
                prediction_horizon=ctrl_params.prediction_horizon,
                step_horizon=ctrl_params.step_horizon,
                n_predictions=ctrl_params.n_predictions,
                n_coincide=ctrl_params.n_coincide,
                scene_builder_cls=TrajectronPlusPlusSceneBuilder,
                turn_choices=scenario_params.turn_choices,
                max_distance=scenario_params.max_distance,
                plot_boundary=False,
                log_agent=False,
                log_cplex=False,
                plot_scenario=True,
                plot_simulation=True,
                plot_overapprox=False,
                scene_config=online_config)
        agent.start_sensor()
        assert agent.sensor_is_listening
        
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
        # configure the spectator
        world.get_spectator().set_transform(
            carla.Transform(
                location,
                # carla.Rotation(pitch=-90, yaw=-90),
                carla.Rotation(pitch=-70, yaw=-90, roll=20)
            )
        )
        record_spectator = False
        if record_spectator:
            # attach camera to spectator
            spectator_camera = attach_camera_to_spectator(world, frame)

        n_burn_frames = scenario_params.n_burn_interval*online_config.record_interval
        if scenario_params.loop_type == LoopEnum.CLOSED_LOOP:
            run_frames = scenario_params.run_interval*online_config.record_interval
        elif isinstance(agent, OAAgent):
            run_frames = ctrl_params.control_horizon*online_config.record_interval - 1
        else:
            run_frames = ctrl_params.n_coincide*online_config.record_interval - 1
        for idx in range(n_burn_frames + run_frames):
            control = None
            for ctrl in scenario_params.controls:
                if ctrl.interval[0] <= idx and idx <= ctrl.interval[1]:
                    control = ctrl.control
                    break
            frame = world.tick()
            agent.run_step(frame, control=control)
 
    finally:
        if spectator_camera:
            spectator_camera.destroy()
        if agent:
            agent.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()
    
        if record_spectator == True:
            time.sleep(5)
        else:
            time.sleep(1)

##################
# Town03 scenarios

CONTROLS_intersection_3 = [
    util.AttrDict(
        interval=(0, 9*5,),
        control=carla.VehicleControl(throttle=0.4)
    ),
]
SCENARIO_intersection_3 = pytest.param(
    # left turn of low curvature to angled road
    ScenarioParameters(
            ego_spawn_idx=85,
            other_spawn_ids=[14],
            # spawn_shifts=[-5, 20],
            spawn_shifts=[-5, 0],
            n_burn_interval=10,
            run_interval=14,
            controls=CONTROLS_intersection_3,
            turn_choices=[1],
            max_distance=75,
            loop_type=LoopEnum.CLOSED_LOOP),
    id="intersection_3"
)
CONTROLS_intersection_3_1 = [
    util.AttrDict(
        interval=(0, 9*5,),
        control=carla.VehicleControl(throttle=0.36)
    ),
]
SCENARIO_intersection_3_1 = pytest.param(
    # left turn of low curvature to angled road
    # 4 other vehicles
    ScenarioParameters(
            ego_spawn_idx=85,
            other_spawn_ids=[14, 14, 15, 15],
            # spawn_shifts=[-5, 31, 23, -11, -19],
            spawn_shifts=[-5, 31, 23, -5, -13],
            n_burn_interval=10,
            run_interval=22,
            controls=CONTROLS_intersection_3_1,
            turn_choices=[1],
            max_distance=100,
            loop_type=LoopEnum.CLOSED_LOOP),
    id="intersection_3_1"
)
CONTROLS_intersection_3_2 = [
    util.AttrDict(
        interval=(0, 9*5,),
        control=carla.VehicleControl(throttle=0.30)
    ),
]
SCENARIO_intersection_3_2 = pytest.param(
    # left turn of low curvature to angled road
    # 4 other vehicles
    ScenarioParameters(
            ego_spawn_idx=85,
            other_spawn_ids=[14, 14, 15, 15],
            spawn_shifts=[-5, 31, 23, -12, -21],
            n_burn_interval=10,
            run_interval=18,
            controls=CONTROLS_intersection_3_2,
            turn_choices=[1],
            max_distance=100,
            loop_type=LoopEnum.CLOSED_LOOP),
    id="intersection_3_2"
)

SCENARIO_intersection_4 = pytest.param(
    # simple left turn
    ScenarioParameters(
            ego_spawn_idx=85,
            other_spawn_ids=[14],
            spawn_shifts=[None, 25],
            n_burn_interval=15,
            run_interval=15,
            controls=CONTROLS_intersection_3,
            turn_choices=[2],
            max_distance=70,
            loop_type=LoopEnum.CLOSED_LOOP),
    id="intersection_4"
)
SCENARIO_intersection_5 = pytest.param(
    # simple right turn
    ScenarioParameters(
            ego_spawn_idx=85,
            other_spawn_ids=[14],
            n_burn_interval=15,
            run_interval=24,
            controls=CONTROLS_intersection_3,
            turn_choices=[3],
            max_distance=70,
            loop_type=LoopEnum.CLOSED_LOOP),
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
            other_spawn_ids=[123, 211],
            n_burn_interval=5,
            run_interval=40,
            controls=CONTROLS_roundabout_1,
            turn_choices=[0, 1],
            max_distance=130,
            loop_type=LoopEnum.CLOSED_LOOP),
    id="roundabout_1"
)

# prediction/control horizon of 8 fails.
# VARIABLES_OAAgent_ph8_ch8_np100

VARIABLES_OAAgent_ph8_ch8_np100 = pytest.param(
    OAAgent, CtrlParameters(
        prediction_horizon=8,
        control_horizon=8,
        n_predictions=100
    ),
    id="OAAgent_ph8_ch8_np100"
)
VARIABLES_OAAgent_ph6_ch6_np200 = pytest.param(
    OAAgent, CtrlParameters(
        prediction_horizon=6,
        control_horizon=6,
        n_predictions=200
    ),
    id="OAAgent_ph6_ch6_np200"
)
VARIABLES_OAAgent_ph6_ch6_np1000 = pytest.param(
    OAAgent, CtrlParameters(
        prediction_horizon=6,
        control_horizon=6,
        n_predictions=1000
    ),
    id="OAAgent_ph6_ch6_np1000"
)
VARIABLES_OAAgent_ph6_ch6_np2000 = pytest.param(
    OAAgent, CtrlParameters(
        prediction_horizon=6,
        control_horizon=6,
        n_predictions=2000
    ),
    id="OAAgent_ph6_ch6_np2000"
)
VARIABLES_OAAgent_ph6_ch3_np100 = pytest.param(
    OAAgent, CtrlParameters(
        prediction_horizon=6,
        control_horizon=3,
        n_predictions=100
    ),
    id="OAAgent_ph6_ch3_np100"
)
VARIABLES_OAAgent_ph6_ch2_np100 = pytest.param(
    OAAgent, CtrlParameters(
        prediction_horizon=6,
        control_horizon=2,
        n_predictions=100
    ),
    id="OAAgent_ph6_ch2_np100"
)
VARIABLES_OAAgent_ph6_ch1_np100 = pytest.param(
    OAAgent, CtrlParameters(
        prediction_horizon=6,
        control_horizon=1,
        n_predictions=100
    ),
    id="OAAgent_ph6_ch1_np100"
)
VARIABLES_MCCAgent_ph6_ch1_np100 = pytest.param(
    MCCAgent, CtrlParameters(
        prediction_horizon=6,
        control_horizon=1,
        n_coincide=2,
        n_predictions=100
    ),
    id="MCCAgent_ph6_ch1_ncoin2_np100"
)
VARIABLES_RMCCAgent_ph6_ch1_np100 = pytest.param(
    RMCCAgent, CtrlParameters(
        prediction_horizon=6,
        control_horizon=1,
        n_coincide=2,
        n_predictions=100
    ),
    id="RMCCAgent_ph6_ch1_ncoin2_np100"
)

@pytest.mark.parametrize(
    "agent_constructor,ctrl_params",
    [
        VARIABLES_OAAgent_ph8_ch8_np100,
        VARIABLES_OAAgent_ph6_ch6_np200,
        VARIABLES_OAAgent_ph6_ch6_np1000,
        VARIABLES_OAAgent_ph6_ch6_np2000,
        VARIABLES_OAAgent_ph6_ch3_np100,
        VARIABLES_OAAgent_ph6_ch2_np100,
        VARIABLES_OAAgent_ph6_ch1_np100,
        VARIABLES_MCCAgent_ph6_ch1_np100,
        VARIABLES_RMCCAgent_ph6_ch1_np100
    ]
)
@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_intersection_3,
        SCENARIO_intersection_3_1,
        SCENARIO_intersection_3_2,
        SCENARIO_intersection_4,
        SCENARIO_intersection_5,
        SCENARIO_roundabout_1
    ]
)
# def test_Town03_scenario(scenario_params, agent_constructor, ctrl_params,
#         carla_Town03_synchronous, eval_env):
def test_Town03_scenario(scenario_params, agent_constructor, ctrl_params,
        carla_Town03_synchronous, eval_env, eval_stg_cuda):
    # eval_stg_cuda = None
    scenario(scenario_params, agent_constructor, ctrl_params,
            carla_Town03_synchronous, eval_env, eval_stg_cuda)
