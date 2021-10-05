import os
import time
import enum
import logging

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import carla
import utility as util
import carlautil

from collect.generate import get_all_vehicle_blueprints
from collect.generate import NaiveMapQuerier
from collect.in_simulation.midlevel.v2_2 import MidlevelAgent as OAAgent
from collect.in_simulation.midlevel.v3 import MidlevelAgent as MCCAgent
from collect.in_simulation.midlevel.v4 import MidlevelAgent as RMCCAgent
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v3_2.trajectron_scene import (
        TrajectronPlusPlusSceneBuilder)

"""Test the v3 midlevel controller

To collect test names call
(broken: starts another instance of CARLA?)
pytest --collect-only

To run tests call
pytest --log-cli-level=INFO --capture=tee-sys
To run one test call e.gl
pytest --log-cli-level=INFO --capture=tee-sys tests/test_in_simulation_v3.py::test_Town03_scenario[ovehicle_turn]
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
        Number of steps to run motion planner. Only applicable to closed loop
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
            # only applicable to closed loop
            run_interval=None,
            controls=[],
            # by default move forward
            goal=None,
            loop_type=LoopEnum.OPEN_LOOP):
        self.ego_spawn_idx = ego_spawn_idx
        self.other_spawn_ids = other_spawn_ids
        self.spawn_shifts = spawn_shifts
        self.n_burn_interval = n_burn_interval
        self.run_interval = run_interval
        self.controls = controls
        self.goal = goal
        self.loop_type = loop_type

class CtrlParameters(object):
    def __init__(self,
            prediction_horizon=8,
            control_horizon=8,
            n_predictions=100,
            # not applicable for OAAgent
            n_coincide=4):
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.n_predictions = n_predictions
        self.n_coincide = n_coincide

def shift_spawn_point(carla_map, k, spawn_shifts, spawn_point):
    try:
        spawn_shift = spawn_shifts[k]
        spawn_shift < 0
    except (TypeError, IndexError) as e:
        return spawn_point
    return carlautil.move_along_road(carla_map, spawn_point, spawn_shift)

def scenario(scenario_params, agent_constructor, ctrl_params,
            carla_synchronous, eval_env, eval_stg):
    ego_vehicle = None
    agent = None
    other_vehicles = []
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
                other_vehicles.append(vehicle)
                other_vehicle_ids.append(vehicle.id)
        
        world.tick()
        agent = agent_constructor(
                ego_vehicle,
                map_reader,
                other_vehicle_ids,
                eval_stg,
                n_burn_interval=scenario_params.n_burn_interval,
                control_horizon=ctrl_params.control_horizon,
                prediction_horizon=ctrl_params.prediction_horizon,
                n_predictions=ctrl_params.n_predictions,
                n_coincide=ctrl_params.n_coincide,
                scene_builder_cls=TrajectronPlusPlusSceneBuilder,
                log_agent=False,
                log_cplex=False,
                plot_scenario=True,
                plot_simulation=True,
                scene_config=online_config)
        agent.start_sensor()
        assert agent.sensor_is_listening
        if scenario_params.goal:
            agent.set_goal(scenario_params.goal.x, scenario_params.goal.y,
                    is_relative=scenario_params.goal.is_relative)
        
        """Move the spectator to the ego vehicle.
        The positioning is a little off"""
        state = agent.get_vehicle_state()
        goal = agent.get_goal()
        if goal.is_relative:
            location = carla.Location(
                    x=state[0] + goal.x,
                    y=state[1] - goal.y,
                    z=state[2] + 50)
        else:
            location = carla.Location(
                    x=state[0] + (state[0] - goal.x) /2.,
                    y=state[1] - (state[1] + goal.y) /2.,
                    z=state[2] + 50)
        world.get_spectator().set_transform(
            carla.Transform(
                location,
                carla.Rotation(pitch=-90)
            )
        )

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
        if agent:
            agent.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()
    
    time.sleep(1)

##################
# Town03 scenarios

CONTROLS_intersection_1 = [
    util.AttrDict(
        interval=(0, 21*5,),
        control=carla.VehicleControl(brake=1.0)
    ),
    util.AttrDict(
        interval=(22*5, 34*5,),
        control=carla.VehicleControl(throttle=0.5)
    )
]
GOAL_intersection_1 = util.AttrDict(
        x=0, y=50, is_relative=True)
SCENARIO_intersection_1 = pytest.param(
    ScenarioParameters(
            ego_spawn_idx=56,
            other_spawn_ids=[241],
            n_burn_interval=35,
            run_interval=15,
            controls=CONTROLS_intersection_1,
            goal=GOAL_intersection_1,
            loop_type=LoopEnum.CLOSED_LOOP),
    id='intersection_1'
)
SCENARIO_intersection_2 = pytest.param(
    ScenarioParameters(
            ego_spawn_idx=56,
            other_spawn_ids=[241, 200],
            spawn_shifts=[None, None, 100],
            n_burn_interval=35,
            run_interval=20,
            controls=CONTROLS_intersection_1,
            goal=GOAL_intersection_1,
            loop_type=LoopEnum.CLOSED_LOOP),
    id='intersection_2'
)

VARIABLES_OAAgent_ph8_ch8_np100 = pytest.param(
    OAAgent, CtrlParameters(
            prediction_horizon=8, control_horizon=8,
            n_predictions=100),
    id='OAAgent_ph8_ch8_np100'
)
VARIABLES_OAAgent_ph4_ch1_np100 = pytest.param(
    OAAgent, CtrlParameters(
            prediction_horizon=4, control_horizon=1,
            n_predictions=100),
    id='OAAgent_ph4_ch1_np100'
)
VARIABLES_MCCAgent_ph4_ch4_np100_ncoin1 = pytest.param(
    MCCAgent, CtrlParameters(
            prediction_horizon=4, control_horizon=4,
            n_predictions=100, n_coincide=1),
    id='MCCAgent_ph4_ch4_np100_ncoin1'
)
VARIABLES_RMCCAgent_ph4_ch4_np100_ncoin1 = pytest.param(
    RMCCAgent, CtrlParameters(
            prediction_horizon=4, control_horizon=4,
            n_predictions=100, n_coincide=1),
    id='RMCCAgent_ph4_ch4_np100_ncoin1'
)

@pytest.mark.parametrize(
    "agent_constructor,ctrl_params",
    [
        VARIABLES_OAAgent_ph8_ch8_np100,
        VARIABLES_OAAgent_ph4_ch1_np100,
        VARIABLES_MCCAgent_ph4_ch4_np100_ncoin1,
        VARIABLES_RMCCAgent_ph4_ch4_np100_ncoin1
    ],
)
@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_intersection_1,
        SCENARIO_intersection_2
    ],
)
def test_Town03_scenario(scenario_params, agent_constructor, ctrl_params,
        carla_Town03_synchronous, eval_env, eval_stg_cuda):
    scenario(scenario_params, agent_constructor, ctrl_params,
            carla_Town03_synchronous, eval_env, eval_stg_cuda)
