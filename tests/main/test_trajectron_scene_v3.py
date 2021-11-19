import os

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import carla
import carlautil

try:
    # trajectron-plus-plus/trajectron
    from environment import Environment, Scene, Node
    from environment import GeometricMap, derivative_of
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from collect.generate import get_all_vehicle_blueprints
from collect.generate import DataCollector
from collect.generate import NaiveMapQuerier, SceneConfig
from collect.generate.scene.v3_2.trajectron_scene import (
        TrajectronPlusPlusSceneBuilder)
from collect.generate.scene.trajectron_util import (
        standardization, plot_trajectron_scene)

TESTSceneBuilder=TrajectronPlusPlusSceneBuilder

"""
pytest --log-cli-level=INFO --capture=tee-sys -vv
"""

def move_along_road(carla_map, transform, distance):
    wp = carla_map.get_waypoint(transform.location,
        project_to_road=True, lane_type=carla.LaneType.Driving)
    if distance < 0: return wp.previous(abs(distance))[0].transform
    else: return wp.next(distance)[0].transform

def shift_spawn_point(carla_map, k, spawn_shifts, spawn_point):
    try:
        spawn_shift = spawn_shifts[k]
        spawn_shift < 0
    except (TypeError, IndexError) as e:
        return spawn_point
    return move_along_road(carla_map, spawn_point, spawn_shift)

def scenario(carla_synchronous, scenario_params):
    n_burn_frames, save_frequency, record_interval, scene_interval, \
            collector_spawn, npc_spawn, spawn_shifts = scenario_params
    client, world, carla_map, traffic_manager = carla_synchronous
    vehicles = []
    data_collectors = []
    env = Environment(node_type_list=['VEHICLE'], standardization=standardization)
    scenes = []
    def add_scene(scene):
        scenes.append(scene)

    try:
        # Mock vehicles
        spawn_points = carla_map.get_spawn_points()
        blueprints = get_all_vehicle_blueprints(world)
        spawn_indices = collector_spawn + npc_spawn
        vehicle_ids = []
        for k, spawn_idx in enumerate(spawn_indices):
            blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[spawn_idx]
            spawn_point = shift_spawn_point(carla_map, k, spawn_shifts, spawn_point)
            # Prevent collision with road.
            spawn_point.location += carla.Location(0, 0, 0.5)
            # carlautil.debug_point(client, spawn_point.location, t=5.0)
            vehicle = world.spawn_actor(blueprint, spawn_point)
            vehicle.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(vehicle)
            vehicle_ids.append(vehicle.id)

        # Setup data collector
        map_reader = NaiveMapQuerier(world, carla_map, debug=True)
        scene_config = SceneConfig(
                scene_interval=scene_interval,
                record_interval=record_interval,
                node_type=env.NodeType)
        n_data_collector = len(collector_spawn)
        for collector_idx in range(n_data_collector):
            other_vehicle_ids = vehicle_ids[:collector_idx] + vehicle_ids[collector_idx + 1:]
            data_collector = DataCollector(
                    vehicles[collector_idx], map_reader,
                    other_vehicle_ids,
                    scene_builder_cls=TESTSceneBuilder,
                    scene_config=scene_config,
                    save_frequency=save_frequency,
                    n_burn_frames=n_burn_frames,
                    callback=add_scene, debug=True)
            data_collector.start_sensor()
            assert data_collector.sensor_is_listening
            data_collectors.append(data_collector)

        # Run simulation for X steps
        for idx in range(n_burn_frames + record_interval*(save_frequency - 1)):
            frame = world.tick()
            for data_collector in data_collectors:
                data_collector.capture_step(frame)
        
        assert len(scenes) == n_data_collector
        for scene in scenes:
            plot_trajectron_scene('out', scene)

    finally:
        for data_collector in data_collectors:
            data_collector.destroy()
        for vehicle in vehicles:
            vehicle.destroy()

##################
# Town03 scenarios

SCENARIO_5_npcs = pytest.param(
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    60, 30, 5, 20, [123], [0, 7, 122, 218, 219], [], id='5_npcs'
)

SCENARIO_2_collectors = pytest.param(
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    40, 40, 5, 30, [7, 187], [122, 221], [], id='2_collectors'
)

SCENARIO_no_npcs = pytest.param(
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    60, 30, 5, 20, [0], [], [], id='no_npcs'
)

SCENARIO_traffic_light = pytest.param(
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    60, 30, 5, 25, [29], [155, 157, 194, 36, 108, 32], [], id='traffic_light'
)

SCENARIO_5_disappearing = pytest.param(
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    40, 30, 5, 20, [126], [263, 264, 262, 127, 128], [], id='5_disappearing'
)

SCENARIO_1_appear = pytest.param(
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    160, 40, 5, 30, [251], [164], [], id='1_appear'
)

##################
# Town04 scenarios

SCENARIO_bridge_1 = pytest.param(
    # 271, 270, 290 are spawn points below bridge
    # 298, 299, 335 are spawn points on the bridge
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    60, 30, 5, 25, [271, 298], [270, 290, 299, 335], [], id='bridge_1'
)


SCENARIO_merge_ramp_1 = pytest.param(
    # 273 spawn start of the ramp from botom
    # 291, 292 spawn points below bridge
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    60, 40, 5, 35, [273, 292], [291, 273], [None, 80, 85, 50], id='merge_ramp_1'
)

SCENARIO_merge_ramp_2 = pytest.param(
    # 306 spawn start of the ramp from top
    # 335, 333 are spawn points on the bridge
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    60, 40, 5, 35, [306, 333], [335, 306], [None, 120, 120, 50], id='merge_ramp_2'
)

##################
# Town05 scenarios

SCENARIO_ramp = pytest.param(
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    60, 40, 5, 30, [265], [], None, id='ramp'
)

SCENARIO_bridge_2 = pytest.param(
    # 265 shift this to top
    # 62, 244 spawn points below the bridge
    # n_burn_frames,save_frequency,record_interval,scene_interval
    # collector_spawn, npc_spawn, spawn_shifts
    60, 30, 5, 25, [265, 62, 244], [], [360, None, None], id='bridge_2'
)

ARGNAMES = (
        "n_burn_frames,save_frequency,record_interval,scene_interval,"
        "collector_spawn,npc_spawn,spawn_shifts")
@pytest.mark.parametrize(
        ARGNAMES, [
            SCENARIO_no_npcs,
            SCENARIO_5_npcs,
            SCENARIO_2_collectors,
            SCENARIO_traffic_light,
            SCENARIO_5_disappearing
        ])
def test_Town03_scenario(carla_Town03_synchronous,
        n_burn_frames, save_frequency, record_interval, scene_interval,
        collector_spawn, npc_spawn, spawn_shifts):
    scenario_params = (n_burn_frames, save_frequency, record_interval, scene_interval, \
            collector_spawn, npc_spawn, spawn_shifts)
    scenario(carla_Town03_synchronous, scenario_params)

@pytest.mark.parametrize(
        ARGNAMES, [
            SCENARIO_bridge_1,
            SCENARIO_merge_ramp_1,
            SCENARIO_merge_ramp_2
        ])
def test_Town04_scenario(carla_Town04_synchronous,
        n_burn_frames, save_frequency, record_interval, scene_interval,
        collector_spawn, npc_spawn, spawn_shifts):
    scenario_params = (n_burn_frames, save_frequency, record_interval, scene_interval, \
            collector_spawn, npc_spawn, spawn_shifts)
    scenario(carla_Town04_synchronous, scenario_params)

@pytest.mark.parametrize(
        ARGNAMES, [
            SCENARIO_ramp,
            SCENARIO_bridge_2
        ])
def test_Town05_scenario(carla_Town05_synchronous,
        n_burn_frames, save_frequency, record_interval, scene_interval,
        collector_spawn, npc_spawn, spawn_shifts):
    scenario_params = (n_burn_frames, save_frequency, record_interval, scene_interval, \
            collector_spawn, npc_spawn, spawn_shifts)
    scenario(carla_Town05_synchronous, scenario_params)
