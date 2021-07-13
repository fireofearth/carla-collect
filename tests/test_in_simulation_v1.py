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
from collect.in_simulation.midlevel.v1 import LCSSHighLevelAgent
from collect.generate.scene import OnlineConfig

"""Test simulation of LCSSHighLevelAgent, the v1 midlevel controller

To collect test names call
(broken: starts another instance of CARLA?)
pytest --collect-only

To run tests call
pytest --log-cli-level=INFO --capture=tee-sys
"""

SCENARIO_straight_road = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,controls,goal
    2, [184, 208], 23, [ ], None,
    id='straight_road'
)

CONTROL_run_straight_road = util.AttrDict(
    interval=(15*5, 22*5,),
    control=carla.VehicleControl(throttle=0.6)
)
SCENARIO_run_straight_road = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,controls,goal
    249, [184, 208], 23, [CONTROL_run_straight_road], None,
    id='run_straight_road'
)

CONTROL_ovehicle_turn = util.AttrDict(
    interval=(67*5, 74*5,),
    control=carla.VehicleControl(throttle=0.6)
)
SCENARIO_ovehicle_turn = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,controls,goal
    87, [164], 76, [CONTROL_ovehicle_turn], None,
    id='ovehicle_turn'
)

CONTROL_merge_lane = util.AttrDict(
    interval=(26*5, 35*5,),
    control=carla.VehicleControl(throttle=0.6)
)
SCENARIO_merge_lane = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,controls,goal
    279, [257], 36, [CONTROL_merge_lane], None,
    id='merge_lane'
)

CONTROL_ego_lane_switch = util.AttrDict(
        interval=(0*5, 19*5,),
        control=carla.VehicleControl(throttle=0.45))
GOAL_ego_lane_switch = util.AttrDict(
        x=50, y=-10, is_relative=True)
SCENARIO_ego_lane_switch_1 = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,controls,goal
    360, [358, 357], 20, [CONTROL_ego_lane_switch], GOAL_ego_lane_switch,
    id='ego_lane_switch_1'
)
SCENARIO_ego_lane_switch_2 = pytest.param(
    # ego_spawn_idx,other_spawn_ids,n_burn_interval,controls,goal
    360, [359, 358], 20, [CONTROL_ego_lane_switch], GOAL_ego_lane_switch,
    id='ego_lane_switch_2'
)

def scenario(params, eval_env, eval_stg):
    ego_spawn_idx, other_spawn_ids, n_burn_interval, controls, \
            goal, carla_synchronous = params
    ego_vehicle = None
    agent = None
    other_vehicles = []
    predict_interval = 8
    prediction_horizon = 8
    n_predictions = 100
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
        agent = LCSSHighLevelAgent(
                ego_vehicle,
                map_reader,
                other_vehicle_ids,
                eval_stg,
                n_burn_interval=n_burn_interval,
                predict_interval=predict_interval,
                prediction_horizon=prediction_horizon,
                n_predictions=n_predictions,
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
        predict_frames = predict_interval*online_config.record_interval - 1
        for idx in range(n_burn_frames + predict_frames):
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
    
    # CARLA has a bug. Tests fail when they are run too
    # quickly after another.
    time.sleep(1)

@pytest.mark.parametrize(
    "ego_spawn_idx,other_spawn_ids,n_burn_interval,controls,goal",
    [
        SCENARIO_straight_road,
        SCENARIO_run_straight_road,
        SCENARIO_ovehicle_turn,
    ],
)
def test_Town03_scenario(ego_spawn_idx, other_spawn_ids, n_burn_interval, controls,
        goal,
        carla_Town03_synchronous, eval_env, eval_stg):
    params = (ego_spawn_idx, other_spawn_ids, n_burn_interval, controls,
            goal, carla_Town03_synchronous)
    scenario(params, eval_env, eval_stg)

@pytest.mark.parametrize(
    "ego_spawn_idx,other_spawn_ids,n_burn_interval,controls,goal",
    [
        SCENARIO_merge_lane,
        SCENARIO_ego_lane_switch_1,
        SCENARIO_ego_lane_switch_2,
    ],
)
def test_Town06_scenario(ego_spawn_idx, other_spawn_ids, n_burn_interval, controls,
        goal,
        carla_Town06_synchronous, eval_env, eval_stg):
    params = (ego_spawn_idx, other_spawn_ids, n_burn_interval, controls,
            goal, carla_Town06_synchronous)
    scenario(params, eval_env, eval_stg)
