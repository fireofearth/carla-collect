import os
import logging

import pytest

import carla

try:
    # imports from trajectron-plus-plus/trajectron
    from environment import Environment, Scene
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from collect.in_simulation import load_model
from collect.generate.scene.v3_2.trajectron_scene import standardization
from . import (
    MODEL_SPEC_1,
    MODEL_SPEC_4,
    MODEL_SPEC_6,
    CARLA_HOST,
    CARLA_PORT,
    SEED
)

model_spec = MODEL_SPEC_6

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
    """Load model in CPU."""
    model_path = os.path.join(os.environ['TRAJECTRONPP_DIR'],
            model_spec.path)
    eval_stg, stg_hyp = load_model(
            model_path, eval_env, ts=20)
    logging.info(model_spec.desc)
    return eval_stg

@pytest.fixture(scope="module")
def eval_stg_cuda(eval_env):
    """Load model in GPU."""
    model_path = os.path.join(os.environ['TRAJECTRONPP_DIR'],
            model_spec.path)
    eval_stg, stg_hyp = load_model(
            model_path, eval_env, ts=20, device='cuda')
    return eval_stg

def instantiate_simulator(map_name):
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    if carla_map.name != map_name:
        world = client.load_world(map_name)
        carla_map = world.get_map()
    traffic_manager = client.get_trafficmanager(8000)
    return client, world, carla_map, traffic_manager

@pytest.fixture(scope="module")
def carla_Town03():
    return instantiate_simulator('Town03')

@pytest.fixture(scope="module")
def carla_Town04():
    return instantiate_simulator('Town04')

@pytest.fixture(scope="module")
def carla_Town05():
    return instantiate_simulator('Town05')

@pytest.fixture(scope="module")
def carla_Town06():
    return instantiate_simulator('Town06')

@pytest.fixture(scope="module")
def carla_Town07():
    return instantiate_simulator('Town07')

@pytest.fixture(scope="module")
def carla_Town10HD():
    return instantiate_simulator('Town10HD')
