
import pytest
import carla

from generate import DataCollector
from generate import NaiveMapQuerier, SceneConfig

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_MAP = 'Town03'
DELTA = 0.1
SEED = 0

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

def test_client(carla_Town03):
    client, world, carla_map, traffic_manager = carla_Town03
    original_settings = None
    ego_vehicle = None
    data_collector = None
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

        # Setup data collector
        map_reader = NaiveMapQuerier(world, carla_map, debug=True)
        scene_config = SceneConfig(scene_interval=10, record_interval=5,
                pixel_dim=0.5)
        data_collector = DataCollector(ego_vehicle,
                map_reader, [],
                scene_config=scene_config,
                save_frequency=20,
                n_burn_frames=1,
                debug=True)
        data_collector.start_sensor()
        assert data_collector.sensor_is_listening

        # Run simulation for X steps
        for idx in range(5 * 19):
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
