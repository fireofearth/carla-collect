import pytest

DELTA = 0.1
SEED = 1

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
    # Note, random seed does not work perfectly
    # traffic_manager.set_random_device_seed(SEED)
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
