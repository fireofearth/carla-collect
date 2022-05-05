import os
import time
import logging
import numpy as np
import pytest

import carla
import utility as util
import carlautil

from tests import (
    LoopEnum,
    ScenarioParameters,
    CtrlParameters,
    shift_spawn_point
)
from collect.exception import CollectException
from collect.generate import get_all_vehicle_blueprints
from collect.generate.scene import OnlineConfig

SCENARIO_straight4 = pytest.param(
    ScenarioParameters(
        other_spawn_ids=[14, 14, 14, 14],
        spawn_shifts=[11, 3, -5, -13],
        other_routes=[
            ["Straight"],
            ["Straight"],
            ["Straight"],
            ["Straight"]
        ],
        run_interval=30,
    ),
    id="straight4"
)

SCENARIO_left4 = pytest.param(
    # OTOH command Right doesn't do anything
    ScenarioParameters(
        other_spawn_ids=[14, 14, 14, 14],
        spawn_shifts=[11, 3, -5, -13],
        other_routes=[
            ["Left"],
            ["Left"],
            ["Left"],
            ["Left"]
        ],
        run_interval=30,
    ),
    id="left4"
)

SCENARIO_right4 = pytest.param(
    ScenarioParameters(
        other_spawn_ids=[15, 15, 15, 15],
        spawn_shifts=[11, 3, -5, -13],
        other_routes=[
            ["Right"],
            ["Right"],
            ["Right"],
            ["Right"]
        ],
        run_interval=30,
    ),
    id="right4"
)


SCENARIO_roundabout4 = pytest.param(
    # paths of OV in a roundabout
    ScenarioParameters(
        other_spawn_ids=[0, 0, 0, 0, 0, 0],
        spawn_shifts=[None, 7, 14, 21, 28, 35],
        other_routes=[
            ["Straight", "Straight"],
            ["Straight", "Straight"],
            ["Straight", "Right"],
            ["Straight", "Right"],
            ["Right"],
            ["Right"],
        ],
        run_interval=50,
    ),
    id="roundabout4"
)

class AutopilotScenario(object):

    def __init__(
        self,
        scenario_params: ScenarioParameters,
        carla_synchronous: tuple,
    ):
        (
            self.client,
            self.world,
            self.carla_map,
            self.traffic_manager
        ) = carla_synchronous
        self.scenario_params = scenario_params

    def run(self):
        other_vehicles = []

        try:
            online_config = OnlineConfig(record_interval=10)

            # Mock vehicles
            spawn_points = self.carla_map.get_spawn_points()
            blueprints = get_all_vehicle_blueprints(self.world)
            spawn_indices = self.scenario_params.other_spawn_ids
            other_vehicle_ids = []
            for k, spawn_idx in enumerate(spawn_indices):
                blueprint = np.random.choice(blueprints)
                spawn_point = spawn_points[spawn_idx]
                spawn_point = shift_spawn_point(
                    self.carla_map, k, self.scenario_params.spawn_shifts, spawn_point
                )
                # Prevent collision with road.
                spawn_point.location += carla.Location(0, 0, 0.5)
                vehicle = self.world.spawn_actor(blueprint, spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                if self.scenario_params.ignore_signs:
                    self.traffic_manager.ignore_signs_percentage(vehicle, 100.)
                if self.scenario_params.ignore_lights:
                    self.traffic_manager.ignore_lights_percentage(vehicle, 100.)
                # if self.scenario_params.ignore_vehicles:
                self.traffic_manager.ignore_vehicles_percentage(vehicle, 0.)
                # if not self.scenario_params.auto_lane_change:
                self.traffic_manager.auto_lane_change(vehicle, False)
                other_vehicles.append(vehicle)
                other_vehicle_ids.append(vehicle.id)

            frame = self.world.tick()
            """Setup vehicle routes"""
            for k, vehicle in enumerate(other_vehicles):
                route = None
                try:
                    route = self.scenario_params.other_routes[k]
                    len(route)
                except (TypeError, IndexError) as e:
                    continue
                self.traffic_manager.set_route(vehicle, route)

            """Move spectator to view all cars."""
            locations = carlautil.to_locations_ndarray(other_vehicles)
            location = carlautil.ndarray_to_location(np.mean(locations, axis=0))
            location += carla.Location(z=50)
            self.world.get_spectator().set_transform(
                carla.Transform(
                    location, carla.Rotation(pitch=-70, yaw=-90, roll=20)
                )
            )

            run_frames = self.scenario_params.run_interval*online_config.record_interval + 1
            for idx in range(run_frames):
                frame = self.world.tick()
                time.sleep(0.02)
        finally:
            for other_vehicle in other_vehicles:
                other_vehicle.destroy()
            time.sleep(1)

@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_straight4,
        SCENARIO_left4,
        SCENARIO_right4,
        SCENARIO_roundabout4,
    ]
)
def test_Town03_scenario(
    scenario_params,
    carla_Town03_synchronous
):
    if "CARLANAME" in os.environ and os.environ["CARLANAME"] == "carla-0.9.13":
        AutopilotScenario(
            scenario_params,
            carla_Town03_synchronous
        ).run()
    else:
        raise CollectException("Autopilot route planning doesn't work for 0.9.11")
