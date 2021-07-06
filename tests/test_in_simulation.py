import os

import dill
import logging
import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import carla

try:
    # imports from trajectron-plus-plus/trajectron
    from environment import Environment, Scene
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from collect.generate import get_all_vehicle_blueprints
from collect.generate import NaiveMapQuerier
from collect.in_simulation import load_model
from collect.in_simulation.midlevel.v1 import LCSSHighLevelAgent
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v2_1.trajectron_scene import (standardization)

"""
pytest --log-cli-level=INFO --capture=tee-sys -vv
"""

@pytest.fixture(scope="module")
def eval_env():
    """Load dummy dataset."""
    eval_scene = Scene(timesteps=25, dt=0.5, name='test')
    eval_env = Environment(node_type_list=['VEHICLE'],
            standardization=standardization)
    attention_radius = dict()
    attention_radius[(eval_env.NodeType.VEHICLE, eval_env.NodeType.VEHICLE)] = 30.0
    eval_env.attention_radius = attention_radius
    eval_env.robot_type = eval_env.NodeType.VEHICLE
    eval_env.scenes = [eval_scene]
    return eval_env

@pytest.fixture(scope="module")
def eval_stg(eval_env):
    """Load model."""
    model_dir = 'experiments/nuScenes/models/20210622'
    model_name = 'models_19_Mar_2021_22_14_19_int_ee_me_ph8'
    model_path = os.path.join(os.environ['TRAJECTRONPP_DIR'],
            model_dir, model_name)
    eval_stg, stg_hyp = load_model(
            model_path, eval_env, ts=20)
    return eval_stg

def test_straight_road(carla_Town03_synchronous, eval_env, eval_stg):
    ego_vehicle = None
    agent = None
    other_vehicles = []
    ego_spawn_idx = 2
    other_spawn_ids = [184, 208]
    n_burn_interval = 23
    predict_interval = 8
    prediction_horizon = 8
    n_predictions = 100
    client, world, carla_map, traffic_manager = carla_Town03_synchronous

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

        n_burn_frames = n_burn_interval*online_config.record_interval
        predict_frames = predict_interval*online_config.record_interval - 1
        for idx in range(n_burn_frames + predict_frames):
            frame = world.tick()
            agent.run_step(frame)
 
    finally:
        if agent:
            agent.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()


def test_ovehicle_turn(carla_Town03_synchronous, eval_env, eval_stg):
    ego_vehicle = None
    agent = None
    other_vehicles = []
    ego_spawn_idx = 87
    other_spawn_ids = [164]
    n_burn_interval = 76
    predict_interval = 8
    prediction_horizon = 8
    n_predictions = 100
    client, world, carla_map, traffic_manager = carla_Town03_synchronous

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

        n_burn_frames = n_burn_interval*online_config.record_interval
        predict_frames = predict_interval*online_config.record_interval - 1
        for idx in range(n_burn_frames + predict_frames):
            frame = world.tick()
            agent.run_step(frame)
 
    finally:
        if agent:
            agent.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()


def test_merge_lane(carla_Town06_synchronous, eval_env, eval_stg):
    ego_vehicle = None
    agent = None
    other_vehicles = []
    ego_spawn_idx = 279
    other_spawn_ids = [257]
    n_burn_interval = 36
    predict_interval = 8
    prediction_horizon = 8
    n_predictions = 100
    client, world, carla_map, traffic_manager = carla_Town06_synchronous

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

        n_burn_frames = n_burn_interval*online_config.record_interval
        predict_frames = predict_interval*online_config.record_interval - 1
        for idx in range(n_burn_frames + predict_frames):
            frame = world.tick()
            agent.run_step(frame)
 
    finally:
        if agent:
            agent.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()

