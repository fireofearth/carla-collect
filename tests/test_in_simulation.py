import os

import dill
import logging
import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import carla

from generate import get_all_vehicle_blueprints
from generate import NaiveMapQuerier
from in_simulation import LCSSHighLevelAgent, load_model
from generate.scene import OnlineConfig

"""
pytest --log-cli-level=INFO --capture=tee-sys -vv
"""

def test_straight_road(carla_Town03_synchronous):
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

    """Load dummy dataset."""
    dataset_path = 'carla_v2_1_dataset/carla_test_v2_1_full.pkl'
    with open(dataset_path, 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')

    """Load model."""
    model_dir = 'experiments/nuScenes/models/20210622'
    model_name = 'models_19_Mar_2021_22_14_19_int_ee_me_ph8'
    model_path = os.path.join(os.environ['TRAJECTRONPP_DIR'],
            model_dir, model_name)
    eval_stg, stg_hyp = load_model(
            model_path, eval_env, ts=20)

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