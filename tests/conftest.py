import os
import logging

import pytest
import numpy as np

import utility as util
import carla

try:
    # imports from trajectron-plus-plus/trajectron
    from environment import Environment, Scene
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from collect.in_simulation import load_model
from collect.generate.scene.v2_1.trajectron_scene import standardization

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_MAP = 'Town03'
DELTA = 0.1
SEED = 1

# os.environ['TRAJECTRONPP_DIR']
model_spec_1 = util.AttrDict(
        path='experiments/nuScenes/models/20210621/models_19_Mar_2021_22_14_19_int_ee_me_ph8',
        desc="Base +Dynamics, Map off-the-shelf model trained on NuScenes")

model_spec_2 = util.AttrDict(
        path='experiments/nuScenes/models/models_20_Jul_2021_11_48_11_carla_v3_0_1_base_distmap_ph8',
        desc="Base +Map model w/ heading fix trained on small CARLA synthesized")

model_spec_3 = util.AttrDict(
        path='experiments/nuScenes/models/20210803/models_03_Aug_2021_13_42_51_carla_v3-1-1_base_distmapV4_ph8',
        desc="Base +MapV4-1 model with heading fix, PH=8, K=25 "
             "(trained on smaller carla v3-1-1 dataset)")

model_spec_4 = util.AttrDict(
        path='experiments/nuScenes/models/20210816/models_17_Aug_2021_13_25_38_carla_v3-1-2_base_distmapV4_modfm_K15_ph8',
        desc="Base +MapV4 model with heading fix, PH=8, K=15 "
             "(trained on carla v3-1-2 dataset)")

model_spec = model_spec_4

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
    traffic_manager.set_random_device_seed(2)
    world.apply_settings(settings)
    return carla_Town

@pytest.fixture(scope="module")
def carla_Town03_synchronous(request, carla_Town03):
    return instantiate_synchronous(request, carla_Town03)

@pytest.fixture(scope="module")
def carla_Town04_synchronous(request, carla_Town04):
    return instantiate_synchronous(request, carla_Town04)

@pytest.fixture(scope="module")
def carla_Town05_synchronous(request, carla_Town05):
    return instantiate_synchronous(request, carla_Town05)

@pytest.fixture(scope="module")
def carla_Town06_synchronous(request, carla_Town06):
    return instantiate_synchronous(request, carla_Town06)

@pytest.fixture(scope="module")
def carla_Town07_synchronous(request, carla_Town07):
    return instantiate_synchronous(request, carla_Town07)

@pytest.fixture(scope="module")
def carla_Town10HD_synchronous(request, carla_Town10HD):
    return instantiate_synchronous(request, carla_Town10HD)
