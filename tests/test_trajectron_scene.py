import os

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import carla

try:
    # trajectron-plus-plus/trajectron
    from environment import Environment, Scene, Node
    from environment import GeometricMap, derivative_of
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from generate import get_all_vehicle_blueprints
from generate import DataCollector
from generate import NaiveMapQuerier, SceneConfig
from generate import TrajectronPlusPlusSceneBuilder

TESTSceneBuilder=TrajectronPlusPlusSceneBuilder

standardization = {
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}

"""
pytest --log-cli-level=INFO -vv -s
"""

def test_no_npcs(carla_Town03_synchronous):
    client, world, carla_map, traffic_manager = carla_Town03_synchronous
    ego_vehicle = None
    data_collector = None
    env = Environment(node_type_list=['VEHICLE'], standardization=standardization)
    scenes = []
    def add_scene(scene):
        scenes.append(scene)

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
        scene_config = SceneConfig(
                scene_interval=scene_interval,
                record_interval=record_interval,
                node_type=env.NodeType)
        data_collector = DataCollector(ego_vehicle,
                map_reader, [],
                scene_builder_cls=TESTSceneBuilder,
                scene_config=scene_config,
                save_frequency=save_frequency,
                n_burn_frames=n_burn_frames,
                callback=add_scene,
                debug=True)
        data_collector.start_sensor()
        assert data_collector.sensor_is_listening

        # Run simulation for X steps
        for idx in range(n_burn_frames + record_interval*(save_frequency - 1)):
            frame = world.tick()
            data_collector.capture_step(frame)
        
        #
        assert len(scenes) == 1

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
    env = Environment(node_type_list=['VEHICLE'], standardization=standardization)
    scenes = []
    def add_scene(scene):
        scenes.append(scene)

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
        scene_config = SceneConfig(
                scene_interval=scene_interval,
                record_interval=record_interval,
                node_type=env.NodeType)
        data_collector = DataCollector(ego_vehicle,
                map_reader, other_vehicle_ids,
                scene_builder_cls=TESTSceneBuilder,
                scene_config=scene_config,
                save_frequency=save_frequency,
                n_burn_frames=n_burn_frames,
                callback=add_scene,
                debug=True)
        data_collector.start_sensor()
        assert data_collector.sensor_is_listening

        # Run simulation for X steps
        for idx in range(n_burn_frames + record_interval*(save_frequency - 1)):
            frame = world.tick()
            data_collector.capture_step(frame)
        
        assert len(scenes) == 1
        scene = scenes[0]
        fig, ax = plt.subplots(figsize=(15,15))
        # extent = (scene.x_min, scene.x_max, scene.y_min, scene.y_max)
        ax.imshow(scene.map['VEHICLE'].as_image(), origin='lower')
        spectral = cm.nipy_spectral(np.linspace(0, 1, len(scene.nodes)))
        for idx, node in enumerate(scene.nodes):
            # using values from scene.map['VEHICLE'].homography
            # to scale points
            xy = 3 * node.data.data[:, :2]
            ax.scatter(xy[:, 0], xy[:, 1], color=spectral[idx])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fig.tight_layout()
        fn = f"{ scene.name.replace('/', '_') }.png"
        fp = os.path.join('out', fn)
        fig.savefig(fp)

    finally:
        if data_collector:
            data_collector.destroy()
        if ego_vehicle:
            ego_vehicle.destroy()
        for other_vehicle in other_vehicles:
            other_vehicle.destroy()
