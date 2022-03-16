import os
import logging

import pytest

import pytest
import numpy as np

import utility as util
import carla

try:
    # imports from trajectron-plus-plus/trajectron
    from environment import Environment, Scene
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from tests import MODEL_SPEC_4, TRAJECTRONPP_DIR
from collect.in_simulation import load_model
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v2_1.trajectron_scene import standardization

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_MAP = 'Town03'
TRAFFICMANAGER_PORT = 8000
DELTA = 0.05

model_spec = MODEL_SPEC_4

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
def online_config(eval_env):
    """Load online config for data collection, specifying to record
    data every 10 steps of simulator."""
    return OnlineConfig(record_interval=10, node_type=eval_env.NodeType)

@pytest.fixture(scope="module")
def eval_stg(eval_env):
    """Load model in CPU."""
    model_path = os.path.join(TRAJECTRONPP_DIR, model_spec.path)
    eval_stg, _ = load_model(model_path, eval_env, ts=20)
    logging.info(model_spec.desc)
    return eval_stg

@pytest.fixture(scope="module")
def eval_stg_cuda(eval_env):
    """Load model in GPU."""
    model_path = os.path.join(TRAJECTRONPP_DIR, model_spec.path)
    eval_stg, _ = load_model(model_path, eval_env, ts=20, device='cuda')
    return eval_stg

def instantiate_simulator(map_name):
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    if carla_map.name != map_name:
        world = client.load_world(map_name)
        carla_map = world.get_map()
    traffic_manager = client.get_trafficmanager(TRAFFICMANAGER_PORT)
    return client, world, carla_map, traffic_manager

@pytest.fixture(scope="module")
def carla_Town03():
    return instantiate_simulator('Town03')

def instantiate_synchronous(request, carla_Town):
    client, world, carla_map, traffic_manager = carla_Town
    original_settings = None
    def tear_down():
        if original_settings:
            world.apply_settings(original_settings)
    request.addfinalizer(tear_down)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = DELTA
    settings.synchronous_mode = True
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    traffic_manager.global_percentage_speed_difference(0.0)
    world.apply_settings(settings)
    return carla_Town

@pytest.fixture(scope="module")
def carla_Town03_synchronous(request, carla_Town03):
    return instantiate_synchronous(request, carla_Town03)
