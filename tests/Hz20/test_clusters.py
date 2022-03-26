import time

import carla
import utility as util

import time
import logging
import numpy as np
import pytest
import carla
import utility as util
import carlautil

try:
    # imports from trajectron-plus-plus/trajectron
    from environment import Environment, Scene
    from model import Trajectron
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from tests import (
    LoopEnum,
    ScenarioParameters,
    CtrlParameters,
    shift_spawn_point
)
from collect.generate import get_all_vehicle_blueprints
from collect.generate import NaiveMapQuerier
from collect.generate.scene import OnlineConfig
from collect.in_simulation.capture.v1 import CapturingAgent
from collect.generate.scene.v3_2.trajectron_scene import (
    TrajectronPlusPlusSceneBuilder
)


SCENARIO_straight4 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=14,
        other_spawn_ids=[14, 14, 14],
        spawn_shifts=[-5, 11, 3, -13],
        other_routes=[
            ["Straight"],
            ["Straight"],
            ["Straight"],
            ["Straight"]
        ],
        n_burn_interval=10,
        run_interval=20,
    ),
    id="straight4"
)

VARIABLES_ph8_np1000 = pytest.param(
    CtrlParameters(
        prediction_horizon=8,
        n_predictions=1000
    ),
    id="ph8_np1000"
)

VARIABLES_ph8_np5000 = pytest.param(
    CtrlParameters(
        prediction_horizon=8,
        n_predictions=5000
    ),
    id="ph8_np5000"
)

class ClusterScenario(object):
    def __init__(
        self,
        scenario_params: ScenarioParameters,
        ctrl_params: CtrlParameters,
        carla_synchronous: tuple,
        eval_env: Environment,
        eval_stg: Trajectron
    ):
        self.client, self.world, self.carla_map, self.traffic_manager = carla_synchronous
        self.scenario_params = scenario_params
        self.ctrl_params = ctrl_params
        self.eval_env = eval_env
        self.eval_stg = eval_stg
        self.scene_builder_cls = TrajectronPlusPlusSceneBuilder
    
    def run(self):
        ego_vehicle = None
        agent = None
        other_vehicles = []

        try:
            map_reader = NaiveMapQuerier(self.world, self.carla_map, debug=True)
            online_config = OnlineConfig(record_interval=10, node_type=self.eval_env.NodeType)

            # Mock vehicles
            spawn_points = self.carla_map.get_spawn_points()
            blueprints = get_all_vehicle_blueprints(self.world)
            spawn_indices = [self.scenario_params.ego_spawn_idx] + self.scenario_params.other_spawn_ids
            other_vehicle_ids = []
            for k, spawn_idx in enumerate(spawn_indices):
                if k == 0:
                    blueprint = self.world.get_blueprint_library().find('vehicle.audi.a2')
                else:
                    blueprint = np.random.choice(blueprints)
                spawn_point = spawn_points[spawn_idx]
                spawn_point = shift_spawn_point(
                    self.carla_map, k, self.scenario_params.spawn_shifts, spawn_point
                )
                # Prevent collision with road.
                spawn_point.location += carla.Location(0, 0, 0.5)
                vehicle = self.world.spawn_actor(blueprint, spawn_point)
                if k == 0:
                    ego_vehicle = vehicle
                else:
                    other_vehicles.append(vehicle)
                    other_vehicle_ids.append(vehicle.id)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                if self.scenario_params.ignore_signs:
                    self.traffic_manager.ignore_signs_percentage(vehicle, 100.)
                if self.scenario_params.ignore_lights:
                    self.traffic_manager.ignore_lights_percentage(vehicle, 100.)
                # if self.scenario_params.ignore_vehicles:
                #     self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.)
                if not self.scenario_params.auto_lane_change:
                    self.traffic_manager.auto_lane_change(vehicle, False)
            
            # Set up data collecting vehicle.
            frame = self.world.tick()
            agent = CapturingAgent(
                ego_vehicle,
                map_reader,
                other_vehicle_ids,
                self.eval_stg,
                scene_builder_cls=self.scene_builder_cls,
                scene_config=online_config,
                **self.scenario_params,
                **self.ctrl_params
            )
            agent.start_sensor()
            assert agent.sensor_is_listening

            # Set up autopilot routes
            for k, vehicle in enumerate([ego_vehicle] + other_vehicles):
                route = None
                try:
                    route = self.scenario_params.other_routes[k]
                    len(route)
                except (TypeError, IndexError):
                    continue
                try:
                    self.traffic_manager.set_route(vehicle, route)
                except AttributeError:
                    break

            locations = carlautil.to_locations_ndarray(other_vehicles)
            location = carlautil.ndarray_to_location(np.mean(locations, axis=0))
            location += carla.Location(z=50)
            self.world.get_spectator().set_transform(
                carla.Transform(
                    location, carla.Rotation(pitch=-70, yaw=-90, roll=20)
                )
            )

            n_burn_frames = self.scenario_params.n_burn_interval*online_config.record_interval
            run_frames = self.scenario_params.run_interval*online_config.record_interval
            for idx in range(n_burn_frames + run_frames):
                frame = self.world.tick()
                agent.run_step(frame)
        
        finally:
            if agent:
                agent.destroy()
            if ego_vehicle:
                ego_vehicle.destroy()
            for other_vehicle in other_vehicles:
                other_vehicle.destroy()
            time.sleep(1)

@pytest.mark.parametrize(
    "ctrl_params",
    [
        VARIABLES_ph8_np1000,
        VARIABLES_ph8_np5000
    ]
)
@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_straight4
    ]
)
def test_Town03_scenario(
    scenario_params, ctrl_params, carla_Town03_synchronous, eval_env, eval_stg_cuda
):
    ClusterScenario(
        scenario_params,
        ctrl_params,
        carla_Town03_synchronous,
        eval_env,
        eval_stg_cuda
    ).run()
