
import pytest
import carla

from generate import DataCollector
from generate import NaiveMapQuerier, SceneConfig

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_MAP = 'Town03'
DELTA = 0.1
SEED = 1

@pytest.fixture
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

def carla_Town03_synchronous(carla_Town03):
    client, world, carla_map, traffic_manager = carla_Town03
    original_settings = None
    ego_vehicle = None
    data_collector = None

"""
pytest --log-cli-level=INFO -vv -s
"""

def test_no_npcs(carla_Town03):
    client, world, carla_map, traffic_manager = carla_Town03
    original_settings = None
    ego_vehicle = None
    data_collector = None

    n_burn_frames = 60
    save_frequency = 30
    record_interval = 5
    scene_interval = 20
    try:
        # Mock DataGenerator.run()
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.fixed_delta_seconds = DELTA
        settings.synchronous_mode = True
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.global_percentage_speed_difference(0.0)
        traffic_manager.set_random_device_seed(SEED)
        world.apply_settings(settings)

        # Mock vehicles
        blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
        spawn_point = carla_map.get_spawn_points()[0]
        ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        # Setup data collector
        map_reader = NaiveMapQuerier(world, carla_map, debug=True)
        scene_config = SceneConfig(scene_interval=scene_interval, record_interval=record_interval,
                pixel_dim=0.5)
        data_collector = DataCollector(ego_vehicle,
                map_reader, [],
                scene_config=scene_config,
                save_frequency=save_frequency,
                n_burn_frames=n_burn_frames,
                debug=True)
        data_collector.start_sensor()
        assert data_collector.sensor_is_listening

        # Run simulation for X steps
        for idx in range(n_burn_frames + record_interval*(save_frequency - 1)):
            frame = world.tick()
            data_collector.capture_step(frame)

    finally:
        # Cleanup
        if original_settings:
            world.apply_settings(original_settings)
        if data_collector:
            data_collector.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
