import os
import time
import logging

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import carla

import utility as util

from collect.generate import get_all_vehicle_blueprints
from collect.generate import NaiveMapQuerier
from collect.in_simulation.midlevel.v2_1 import MidlevelAgent
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v2_2.trajectron_scene import (
        TrajectronPlusPlusSceneBuilder)

"""Test the v2 midlevel controller

To collect test names call
(broken: starts another instance of CARLA?)
pytest --collect-only

To run tests call
pytest --log-cli-level=INFO --capture=tee-sys
To run one test call e.gl
pytest --log-cli-level=INFO --capture=tee-sys tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn-ph4_ch1_np100]
"""

CONTROL_ovehicle_turn = util.AttrDict(
    interval=(69*5, 76*5,),
    control=carla.VehicleControl(throttle=0.6)
)
SCENARIO_ovehicle_turn = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,run_interval,controls,goal
    87, [164], 77, 15, [CONTROL_ovehicle_turn], None,
    id='ovehicle_turn'
)
SCENARIO_ovehicle_turn_short = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,run_interval,controls,goal
    87, [164], 77, 4, [CONTROL_ovehicle_turn], None,
    id='ovehicle_turn_short'
)

CONTROL_merge_lane = util.AttrDict(
    interval=(26*5, 35*5,),
    control=carla.VehicleControl(throttle=0.6)
)
SCENARIO_merge_lane = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,run_interval,controls,goal
    279, [257], 36, 15, [CONTROL_merge_lane], None,
    id='merge_lane'
)
SCENARIO_merge_lane_short = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,run_interval,controls,goal
    279, [257], 36, 8, [CONTROL_merge_lane], None,
    id='merge_lane_short'
)

VARIABLES_ph8_ch8_np100 = pytest.param(
    # prediction_horizon,control_horizon,n_predictions
    8, 8, 100,
    id='ph8_ch8_np100'
)
VARIABLES_ph8_ch8_np5000 = pytest.param(
    # prediction_horizon,control_horizon,n_predictions
    8, 8, 100,
    id='ph8_ch8_np5000'
)
VARIABLES_ph4_ch1_np5000 = pytest.param(
    # prediction_horizon,control_horizon,n_predictions
    4, 1, 5000,
    id='ph4_ch1_np5000'
)
VARIABLES_ph4_ch1_np100 = pytest.param(
    # prediction_horizon,control_horizon,n_predictions
    4, 1, 100,
    id='ph4_ch1_np100'
)

def scenario(scenario_params, variables, eval_env, eval_stg):
    ego_spawn_idx, other_spawn_ids, n_burn_interval, run_interval, \
            controls, goal, carla_synchronous = scenario_params
    prediction_horizon, control_horizon, n_predictions = variables
    ego_vehicle = None
    agent = None
    other_vehicles = []
    client, world, carla_map, traffic_manager = carla_synchronous

    try:
        map_reader = NaiveMapQuerier(
                world, carla_map, debug=True)
        online_config = OnlineConfig(node_type=eval_env.NodeType)

        spawn_points = carla_map.get_spawn_points()
        spawn_point = spawn_points[ego_spawn_idx]
        blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
        ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        
        other_vehicle_ids = []
        blueprints = get_all_vehicle_blueprints(world)
        for idx in other_spawn_ids:
            blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[idx]
            other_vehicle = world.spawn_actor(blueprint, spawn_point)
            other_vehicle.set_autopilot(True, traffic_manager.get_port())
            other_vehicles.append(other_vehicle)
            other_vehicle_ids.append(other_vehicle.id)
        
        world.tick()
        agent = MidlevelAgent(
                ego_vehicle,
                map_reader,
                other_vehicle_ids,
                eval_stg,
                n_burn_interval=n_burn_interval,
                control_horizon=control_horizon,
                prediction_horizon=prediction_horizon,
                n_predictions=n_predictions,
                scene_builder_cls=TrajectronPlusPlusSceneBuilder,
                scene_config=online_config)
        agent.start_sensor()
        assert agent.sensor_is_listening
        if goal:
            agent.set_goal(goal.x, goal.y, is_relative=True)
        
        """Move the spectator to the ego vehicle.
        The positioning is a little off"""
        state = agent.get_vehicle_state()
        goal = agent.get_goal()
        world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(
                    x=state[0] + goal.x,
                    y=state[1] - goal.y,
                    z=state[2] + 50
                ),
                carla.Rotation(pitch=-90)
            )
        )

        n_burn_frames = n_burn_interval*online_config.record_interval
        run_frames = run_interval*online_config.record_interval
        for idx in range(n_burn_frames + run_frames):
            control = None
            for ctrl in controls:
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

@pytest.mark.parametrize(
    "prediction_horizon,control_horizon,n_predictions",
    [
        VARIABLES_ph8_ch8_np100,
        VARIABLES_ph4_ch1_np100,
        VARIABLES_ph4_ch1_np5000,
    ],
)
@pytest.mark.parametrize(
    "ego_spawn_idx,other_spawn_ids,n_burn_interval,run_interval,controls,goal",
    [
        SCENARIO_ovehicle_turn,
        SCENARIO_ovehicle_turn_short,
    ],
)
def test_Town03_scenario(ego_spawn_idx, other_spawn_ids,
        n_burn_interval, run_interval, controls, goal,
        prediction_horizon, control_horizon, n_predictions,
        carla_Town03_synchronous, eval_env, eval_stg_cuda):
    scenario_params = (ego_spawn_idx, other_spawn_ids,
            n_burn_interval, run_interval,
            controls, goal, carla_Town03_synchronous)
    variables = (prediction_horizon, control_horizon, n_predictions)
    scenario(scenario_params, variables, eval_env, eval_stg_cuda)

@pytest.mark.parametrize(
    "prediction_horizon,control_horizon,n_predictions",
    [
        VARIABLES_ph8_ch8_np100,
        VARIABLES_ph8_ch8_np5000,
        VARIABLES_ph4_ch1_np100,
        VARIABLES_ph4_ch1_np5000,
    ],
)
@pytest.mark.parametrize(
    "ego_spawn_idx,other_spawn_ids,n_burn_interval,run_interval,controls,goal",
    [
        SCENARIO_merge_lane,
        SCENARIO_merge_lane_short,
    ],
)
def test_Town06_scenario(ego_spawn_idx, other_spawn_ids,
        n_burn_interval, run_interval, controls, goal,
        prediction_horizon, control_horizon, n_predictions,
        carla_Town06_synchronous, eval_env, eval_stg_cuda):
    scenario_params = (ego_spawn_idx, other_spawn_ids,
            n_burn_interval, run_interval,
            controls, goal, carla_Town06_synchronous)
    variables = (prediction_horizon, control_horizon, n_predictions)
    scenario(scenario_params, variables, eval_env, eval_stg_cuda)
