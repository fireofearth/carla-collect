import os
import logging
import math

import pytest
import numpy as np
import matplotlib.pyplot as plt

import utility as util
import carla
import carlautil

"""
pytest tests/20Hz/test_api.py::test_Town03_scenario[forward]
"""

class ScenarioParameters(object):
    """Scenario parameters.

    Attributes
    ==========
    ego_spawn_idx : int
        Index of spawn location for EV.
    spawn_shift : float or None
        Amount in meters to shift from spawn location along the road.
    run_steps : int
        Number of simulator steps to run motion planner.
    controls : carla.AttrDict
        The controls to send to PID controller.
    """

    def __init__(self,
            ego_spawn_idx=None,
            spawn_shift=None,
            run_steps=None,
            controls=None
    ):
        self.ego_spawn_idx = ego_spawn_idx
        self.spawn_shift = spawn_shift
        self.run_steps = run_steps
        self.controls = controls


def scenario(scenario_params, carla_synchronous):
    client, world, carla_map, traffic_manager = carla_synchronous
    ego_vehicle = None

    try:
        # Set up the vehicle
        spawn_points = carla_map.get_spawn_points()
        spawn_point = spawn_points[scenario_params.ego_spawn_idx]
        if scenario_params.spawn_shift is not None:
            spawn_point = carlautil.move_along_road(
                    carla_map, spawn_point, scenario_params.spawn_shift)
        spawn_point.location += carla.Location(0, 0, 0.1)
        blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
        ego_vehicle = world.spawn_actor(blueprint, spawn_point)
        world.tick()
    
        # Set up the camera.
        shift = 45
        shift = shift*ego_vehicle.get_transform().get_forward_vector()
        shift = shift + carla.Location(z=65)
        location = ego_vehicle.get_transform().location + shift
        world.get_spectator().set_transform(
            carla.Transform(
                location,
                carla.Rotation(pitch=-90)
            )
        )
        

        # NOTE: ego_vehicle.get_speed_limit() documentation says m/s. This is incorrect.
        logging.info(f"Vehicle speed limit is {ego_vehicle.get_speed_limit()} km/h")

        ###############
        # Apply control
        # NOTE: 1 m/s == 3.6 km/h
        speeds = []
        accelerations = []
        vangulars = []
        control_throttles = []
        control_brakes = []
        control_steerings = []
        for idx in range(scenario_params.run_steps):
            for ctrl in scenario_params.controls:
                if ctrl.interval[0] <= idx and idx <= ctrl.interval[1]:
                    ego_vehicle.apply_control(ctrl.control)
                    break

            world.tick()
            speed = carlautil.actor_to_speed(ego_vehicle)
            speeds.append(speed)
            accel = carlautil.actor_to_acceleration(ego_vehicle)
            accelerations.append(accel)
            vangular = ego_vehicle.get_angular_velocity().z
            vangulars.append(vangular)
            control_throttles.append(ctrl.control.throttle)
            control_brakes.append(ctrl.control.brake)
            control_steerings.append(ctrl.control.steer)

        ##########
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(12, 12))
        axes = axes.ravel()
        timesteps = np.arange(len(speeds)) * 0.05
        axes[0].plot(timesteps, speeds, label="measurement")
        axes[0].set_title("Speed input/response")
        axes[0].set_ylabel("$m/s$")
        axes[1].plot(timesteps, accelerations, label="measurement")
        axes[1].set_title("Acceleration response")
        axes[1].set_ylabel("$m/s^2$")
        axes[2].plot(timesteps, vangulars, label="measurement")
        axes[2].plot(timesteps, control_steerings, label="steer input")
        axes[2].set_title("Angular velocity response")
        axes[2].set_ylabel("deg/$s$")
        for ax in axes:
            ax.legend()
            ax.set_xlabel("time s")
        for ax in axes[:2]:
            ax.plot(timesteps, control_throttles, "b", label="throttle input")
            ax.plot(timesteps, control_brakes, "r", label="brake input")
        plt.show()

    finally:
        if ego_vehicle:
            ego_vehicle.destroy()


"""
Go straight at top speed

- Intersection speed limit is 30 km/h == 8.33.. m/s
- Setting throttle to 1 makes car velocity increase endlessly(?),
  when using automatic gear. This is not always true when throttle is <1.
- Audi A2 reaches 8 m/s within 2 seconds.
- Car uses gear shifts so going from 0 to high speed requires gear shifts,
  and hence variable accelerations.
- The higher the gear, the lower the acceleration.

Gear -1 - reverse gear, move backwards
Gear 0 - set to neutral gear
The “N” is an indicator that your automatic transmission
is in NEUTRAL or a free spinning mode.
Gear 1 - up to 5 m/s^2, up to 8 m/s
Gear 2 - up to 4 m/s^2, up to 14 m/s
Gear 3 - up to 3 m/s^2, up to 20 m/s
"""
SCENARIO_forward = pytest.param(
    ScenarioParameters(
        # ego_spawn_idx=85,
        ego_spawn_idx=264,
        spawn_shift=None,
        run_steps=200,
        controls=[
            util.AttrDict(
                interval=(0, 200),
                control=carla.VehicleControl(throttle=1.0)
            )
        ]
    ),
    id="forward"
)

SCENARIO_brake = pytest.param(
    # Go straight, then stop
    ScenarioParameters(
        ego_spawn_idx=264,
        spawn_shift=None,
        run_steps=200,
        controls=[
            util.AttrDict(
                interval=(0, 99),
                control=carla.VehicleControl(throttle=1.0)
            ),
            util.AttrDict(
                interval=(100, 200),
                control=carla.VehicleControl(brake=1.0)
            )
        ]
    ),
    id="brake"
)

SCENARIO_go_stop_go = pytest.param(
    # Go straight, then stop, then go
    ScenarioParameters(
        ego_spawn_idx=264,
        spawn_shift=None,
        run_steps=150,
        controls=[
            util.AttrDict(
                interval=(0, 49),
                control=carla.VehicleControl(throttle=1.0)
            ),
            util.AttrDict(
                interval=(50, 99),
                control=carla.VehicleControl(brake=1.0)
            ),
            util.AttrDict(
                interval=(100, 150),
                control=carla.VehicleControl(throttle=1.0)
            )
        ]
    ),
    id="go_stop_go"
)

"""
Go straight, then stop, then go.
manual_gear_shift=True and gear=1 must be set in control
at all times or else gear will start changing. 
When fixing gear 1,
The vehicle is able to keep up an acceleration of > 1.0 m/s^2
until it reaches a velocity of 8 m/s.
"""
SCENARIO_gear_1_go_stop_go = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=264,
        spawn_shift=None,
        run_steps=150,
        controls=[
            util.AttrDict(
                interval=(0, 49),
                control=carla.VehicleControl(
                    throttle=1.0,
                    gear=1,
                    manual_gear_shift=True
                )
            ),
            util.AttrDict(
                interval=(50, 99),
                control=carla.VehicleControl(
                    brake=1.0,
                    gear=1,
                    manual_gear_shift=True
                )
            ),
            util.AttrDict(
                interval=(100, 150),
                control=carla.VehicleControl(
                    throttle=1.0,
                    gear=1,
                    manual_gear_shift=True
                )
            )
        ]
    ),
    id="gear_1_go_stop_go"
)

"""
Go straight, increase speed up to 4 m/s and maintain that speed
by updating throttle to 0.42.
"""
SCENARIO_gear_1_to_40ms = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        run_steps=200,
        controls=[
            util.AttrDict(
                interval=(0, 9),
                control=carlautil.create_gear_control(throttle=1.)
            ),
            util.AttrDict(
                interval=(10, 200),
                control=carlautil.create_gear_control(throttle=0.42)
            )
        ]
    ),
    id="gear_1_to_40ms"
)

"""
Also achieve going to speed 4m/s, albeit slowly
by fixing the throttle to 0.42.
"""
SCENARIO_gear_1_to_40ms_slow = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        run_steps=200,
        controls=[
            util.AttrDict(
                interval=(0, 200),
                control=carlautil.create_gear_control(throttle=0.42)
            ),
        ]
    ),
    id="gear_1_to_40ms_slow"
)

"""
At 4m/s (throttle=0.42), when is steering=1 applied then velocity
drops to 3m/s and stays that way. Acceleration jumps and stays at 4 m/s^2.
Angular velocity jumps and stays at to 70 deg/s.
- Steering produces fixed angular velocity
- Throttle at constant gear produces fixed velocity after the initial
  acceleration, and the acceleration is close to zero after.
"""
SCENARIO_gear_1_right_turn10 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        run_steps=200,
        controls=[
            util.AttrDict(
                interval=(0, 9),
                control=carlautil.create_gear_control(throttle=1.)
            ),
            util.AttrDict(
                interval=(10, 89),
                control=carlautil.create_gear_control(throttle=0.42)
            ),
            util.AttrDict(
                interval=(90, 200),
                control=carlautil.create_gear_control(
                    throttle=0.42,
                    steer=1.0
                )
            )
        ]
    ),
    id="gear_1_right_turn10"
)


"""
At 4m/s (throttle=0.42), when is steering=1 applied then velocity
drops to 3.6m/s and stays that way. Acceleration jumps and stays at 3 m/s^2.
Angular velocity jumps and stays at to 42 deg/s.
"""
SCENARIO_gear_1_right_turn05 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        run_steps=200,
        controls=[
            util.AttrDict(
                interval=(0, 9),
                control=carlautil.create_gear_control(throttle=1.)
            ),
            util.AttrDict(
                interval=(10, 119),
                control=carlautil.create_gear_control(throttle=0.42)
            ),
            util.AttrDict(
                interval=(120, 200),
                control=carlautil.create_gear_control(
                    throttle=0.42,
                    steer=0.5
                )
            )
        ]
    ),
    id="gear_1_right_turn05"
)


@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_forward,
        SCENARIO_brake,
        SCENARIO_go_stop_go,
        SCENARIO_gear_1_go_stop_go,
        SCENARIO_gear_1_to_40ms,
        SCENARIO_gear_1_to_40ms_slow,
        SCENARIO_gear_1_right_turn10,
        SCENARIO_gear_1_right_turn05
    ]
)
def test_Town03_scenario(scenario_params, carla_Town03_synchronous):
    scenario(scenario_params, carla_Town03_synchronous)
