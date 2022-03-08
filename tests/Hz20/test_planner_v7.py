import os
import time

import pytest
import numpy as np

import carla
import utility as util
import carlautil

from tests import LoopEnum
from collect.generate import get_all_vehicle_blueprints
from collect.generate import NaiveMapQuerier
from collect.in_simulation.midlevel.v7 import MidlevelAgent
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v3_2.trajectron_scene import (
    TrajectronPlusPlusSceneBuilder
)

"""Test the midlevel controller v7.

pytest tests/20Hz/test_planner_v7.py::test_Town03_scenario[intersection_3-ph6_step1_ncoin1_np100]
"""

class ScenarioParameters(object):
    """
    Attributes
    ==========
    ego_spawn_idx : int
        Index of spawn point to place EV.
    spawn_shifts : number or none
        spawn shifts for the vehicles i=1,2,... in
        `[ego_spawn_idx] + other_spawn_ids`.
        Value of `spawn_shifts[i]` is the distance
        from original spawn point to place vehicle i.
        Let `spawn_shifts[i] = None` to disable shifting.
    n_burn_interval : int
        Number of timesteps before starting motion planning.
    run_interval : int
        Number of steps to run motion planner.
        Only applicable to closed loop.
    controls : list of util.AttrDict
        Optional deterministic controls to apply to vehicle. Each control has
        attributes `interval` that specifies which frames to apply control,
        and `control` containing a carla.VehicleControl to apply to vehicle.
    goal : util.AttrDict
        Optional goal destination the motion planned vehicle should go to.
        By default the vehicle moves forwards.
        Not applicable to curved road segmented boundary constraints.
    turn_choices : list of int
    max_distance : number
    """

    def __init__(self,
            ego_spawn_idx=None,
            other_spawn_ids=[],
            spawn_shifts=[],
            n_burn_interval=None,
            run_interval=None,
            controls=[],
            goal=None,
            turn_choices=[],
            max_distance=100,
            ignore_signs=True,
            ignore_lights=True,
            ignore_vehicles=True,
            auto_lane_change=False,):
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

class CtrlParameters(object):
    def __init__(self,
            n_predictions=100,
            prediction_horizon=8,
            control_horizon=8,
            step_horizon=1,
            n_coincide=1,
            random_mcc=False,
            loop_type=LoopEnum.OPEN_LOOP):
        self.n_predictions = n_predictions
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.step_horizon = step_horizon
        self.n_coincide = n_coincide
        self.random_mcc = random_mcc
        self.loop_type = loop_type

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
    sensor = world.spawn_actor(
        blueprint, carla.Transform(), attach_to=world.get_spectator()
    )
    def take_picture(image):
        image.save_to_disk(
            f"out/starting{frame}/frame{image.frame}_spectator.png"
        )
    sensor.listen(take_picture)
    return sensor

def scenario(scenario_params, ctrl_params,
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
        online_config = OnlineConfig(record_interval=10, node_type=eval_env.NodeType)

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
        agent = MidlevelAgent(
                ego_vehicle,
                map_reader,
                other_vehicle_ids,
                eval_stg,
                n_burn_interval=scenario_params.n_burn_interval,
                control_horizon=ctrl_params.control_horizon,
                prediction_horizon=ctrl_params.prediction_horizon,
                step_horizon=ctrl_params.step_horizon,
                n_coincide=ctrl_params.n_coincide,
                n_predictions=ctrl_params.n_predictions,
                random_mcc=ctrl_params.random_mcc,
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
                carla.Rotation(pitch=-70, yaw=-90, roll=20)
            )
        )
        record_spectator = False
        if record_spectator:
            # attach camera to spectator
            spectator_camera = attach_camera_to_spectator(world, frame)

        n_burn_frames = scenario_params.n_burn_interval*online_config.record_interval
        if ctrl_params.loop_type == LoopEnum.CLOSED_LOOP:
            run_frames = scenario_params.run_interval*online_config.record_interval
        else:
            run_frames = ctrl_params.control_horizon*online_config.record_interval - 1
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

@pytest.mark.parametrize(
    "ctrl_params",
    [
        VARIABLES_ph6_step1_ncoin1_np100,
        VARIABLES_ph6_step1_ncoin1_r_np100,
        VARIABLES_ph8_step1_ncoin1_r_np100
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
    scenario(
        scenario_params, ctrl_params,
        carla_Town03_synchronous, eval_env,
        eval_stg_cuda
    )