import pytest
import numpy as np
import carla

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_MAP = 'Town03'
DELTA = 0.1
SEED = 1

@pytest.fixture(scope="module")
def carla_Town03():
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    if carla_map.name != CARLA_MAP:
        world = client.load_world(CARLA_MAP)
        carla_map = world.get_map()
    traffic_manager = client.get_trafficmanager(8000)
    return client, world, carla_map, traffic_manager

@pytest.fixture(scope="module")
def carla_Town03_synchronous(request, carla_Town03):
    client, world, carla_map, traffic_manager = carla_Town03
    original_settings = None
    def tear_down():
        if original_settings:
            world.apply_settings(original_settings)
    request.addfinalizer(tear_down)
    # Mock DataGenerator.run()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = DELTA
    settings.synchronous_mode = True
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    traffic_manager.global_percentage_speed_difference(0.0)
    world.apply_settings(settings)
    return carla_Town03
