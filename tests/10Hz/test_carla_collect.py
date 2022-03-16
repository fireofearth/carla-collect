import pytest
import numpy as np
import carla

from collect.generate import get_all_vehicle_blueprints
from collect.generate import DataCollector
from collect.generate import NaiveMapQuerier, SceneConfig
from collect.generate.scene.misc_scene import (
        BitmapSceneBuilder,
        DistanceTransformSceneBuilder,
        DistanceTransformSelectSceneBuilder,
        DTSelectWithMapSceneBuilder)

TESTSceneBuilder=DTSelectWithMapSceneBuilder

"""
pytest --log-cli-level=INFO --capture=tee-sys -vv
"""

def test_no_npcs(carla_Town03_synchronous):
    client, world, carla_map, traffic_manager = carla_Town03_synchronous
    ego_vehicle = None
    data_collector = None

    n_burn_frames = 60
    save_frequency = 30
    record_interval = 5
    scene_interval = 20
    try:
        # Mock vehicles
        blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
        spawn_point = carla_map.get_spawn_points()[0]
        ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        # Setup data collector
        map_reader = NaiveMapQuerier(world, carla_map, debug=True)
        scene_config = SceneConfig(scene_interval=scene_interval, record_interval=record_interval)
        data_collector = DataCollector(ego_vehicle,
                map_reader, [],
                scene_builder_cls=TESTSceneBuilder,
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
        if data_collector:
            data_collector.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()

def test_5_npcs(carla_Town03_synchronous):
    client, world, carla_map, traffic_manager = carla_Town03_synchronous
    ego_vehicle = None
    other_vehicles = []
    data_collector = None

    n_burn_frames = 60
    save_frequency = 30
    record_interval = 5
    scene_interval = 20
    try:
        # Mock vehicles
        spawn_points = carla_map.get_spawn_points()
        blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
        spawn_point = spawn_points[123]
        ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        other_vehicle_ids = []
        blueprints = get_all_vehicle_blueprints(world)
        for idx in [0, 7, 122, 218, 219]:
            blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[idx]
            other_vehicle = world.spawn_actor(blueprint, spawn_point)
            other_vehicle.set_autopilot(True, traffic_manager.get_port())
            other_vehicles.append(other_vehicle)
            other_vehicle_ids.append(other_vehicle.id)

        # Setup data collector
        map_reader = NaiveMapQuerier(world, carla_map, debug=True)
        scene_config = SceneConfig(scene_interval=scene_interval, record_interval=record_interval)
        data_collector = DataCollector(ego_vehicle,
                map_reader, other_vehicle_ids,
                scene_builder_cls=TESTSceneBuilder,
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
        if data_collector:
            data_collector.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()

def test_not_moving(carla_Town03_synchronous):
    client, world, carla_map, traffic_manager = carla_Town03_synchronous
    ego_vehicle = None
    other_vehicles = []
    data_collector = None

    n_burn_frames = 60
    save_frequency = 30
    record_interval = 5
    scene_interval = 25
    try:
        # Mock vehicles
        spawn_points = carla_map.get_spawn_points()
        blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
        spawn_point = spawn_points[29]
        ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        other_vehicle_ids = []
        blueprints = get_all_vehicle_blueprints(world)
        for idx in [155, 157, 194, 36, 108, 32]:
            blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[idx]
            other_vehicle = world.spawn_actor(blueprint, spawn_point)
            other_vehicle.set_autopilot(True, traffic_manager.get_port())
            other_vehicles.append(other_vehicle)
            other_vehicle_ids.append(other_vehicle.id)

        # Setup data collector
        map_reader = NaiveMapQuerier(world, carla_map, debug=True)
        scene_config = SceneConfig(
                scene_interval=scene_interval,
                record_interval=record_interval)
        data_collector = DataCollector(ego_vehicle,
                map_reader, other_vehicle_ids,
                scene_builder_cls=TESTSceneBuilder,
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
        if data_collector:
            data_collector.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()

def test_5_npcs_disappearing(carla_Town03_synchronous):
    """Test case where NPCs move away from the EGO vehicle and disappear.
    """
    client, world, carla_map, traffic_manager = carla_Town03_synchronous
    ego_vehicle = None
    other_vehicles = []
    data_collector = None

    n_burn_frames = 30
    save_frequency = 30
    record_interval = 5
    scene_interval = 20
    try:
        # Mock vehicles
        spawn_points = carla_map.get_spawn_points()
        blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
        spawn_point = spawn_points[126]
        ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        other_vehicle_ids = []
        blueprints = get_all_vehicle_blueprints(world)
        for idx in [263, 264, 262, 127, 128]:
            blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[idx]
            other_vehicle = world.spawn_actor(blueprint, spawn_point)
            other_vehicle.set_autopilot(True, traffic_manager.get_port())
            other_vehicles.append(other_vehicle)
            other_vehicle_ids.append(other_vehicle.id)

        # Setup data collector
        map_reader = NaiveMapQuerier(world, carla_map, debug=True)
        scene_config = SceneConfig(scene_interval=scene_interval, record_interval=record_interval)
        data_collector = DataCollector(ego_vehicle,
                map_reader, other_vehicle_ids,
                scene_builder_cls=TESTSceneBuilder,
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
        if data_collector:
            data_collector.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()

def test_1_npc_appear(carla_Town03_synchronous):
    """Test case where NPC appears before EGO vehicle after awhile.
    """
    client, world, carla_map, traffic_manager = carla_Town03_synchronous
    ego_vehicle = None
    other_vehicles = []
    data_collector = None

    n_burn_frames = 180
    save_frequency = 40
    record_interval = 5
    scene_interval = 30
    try:
        # Mock vehicles
        spawn_points = carla_map.get_spawn_points()
        blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
        spawn_point = spawn_points[251]
        ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())

        other_vehicle_ids = []
        blueprints = get_all_vehicle_blueprints(world)
        for idx in [164]:
            blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[idx]
            other_vehicle = world.spawn_actor(blueprint, spawn_point)
            other_vehicle.set_autopilot(True, traffic_manager.get_port())
            other_vehicles.append(other_vehicle)
            other_vehicle_ids.append(other_vehicle.id)

        # Setup data collector
        map_reader = NaiveMapQuerier(world, carla_map, debug=True)
        scene_config = SceneConfig(scene_interval=scene_interval, record_interval=record_interval)
        data_collector = DataCollector(ego_vehicle,
                map_reader, other_vehicle_ids,
                scene_builder_cls=TESTSceneBuilder,
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
        if data_collector:
            data_collector.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()

def test_2_data_collectors(carla_Town03_synchronous):
    """Test case where NPCs move away from the EGO vehicle and disappear.
    """
    client, world, carla_map, traffic_manager = carla_Town03_synchronous
    vehicles = []
    data_collectors = []

    n_burn_frames = 40
    save_frequency = 40
    record_interval = 5
    scene_interval = 30
    try:
        # Mock vehicles
        spawn_points = carla_map.get_spawn_points()
        vehicle_ids = []
        blueprints = get_all_vehicle_blueprints(world)
        for idx in [7, 187, 122, 221]:
            blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[idx]
            vehicle = world.spawn_actor(blueprint, spawn_point)
            vehicle.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(vehicle)
            vehicle_ids.append(vehicle.id)

        # Setup data collector
        map_reader = NaiveMapQuerier(world, carla_map, debug=True)
        scene_config = SceneConfig(scene_interval=scene_interval, record_interval=record_interval)
        data_collector = DataCollector(vehicles[0],
                map_reader, vehicle_ids[1:],
                scene_builder_cls=TESTSceneBuilder,
                scene_config=scene_config,
                save_frequency=save_frequency,
                n_burn_frames=n_burn_frames,
                debug=True)
        data_collector.start_sensor()
        data_collectors.append(data_collector)
        data_collector = DataCollector(vehicles[1],
                map_reader, vehicle_ids[:1] + vehicle_ids[2:],
                scene_builder_cls=TESTSceneBuilder,
                scene_config=scene_config,
                save_frequency=save_frequency,
                n_burn_frames=n_burn_frames,
                debug=True)
        data_collector.start_sensor()
        data_collectors.append(data_collector)

        # Run simulation for X steps
        for idx in range(n_burn_frames + record_interval*(save_frequency - 1)):
            frame = world.tick()
            for data_collector in data_collectors:
                data_collector.capture_step(frame)

    finally:
        for data_collector in data_collectors:
            data_collector.destroy()
        for vehicle in vehicles:
            vehicle.destroy()
