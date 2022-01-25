import os
import logging
import math

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import utility as util
import carla
import carlautil

from collect.in_simulation.lowlevel.v3 import VehiclePIDController

"""
pytest tests/20Hz/test_controller_v2.py::test_Town03_scenario[top555]
"""

class ScenarioParameters(object):
    """Scenario parameters.

    Attributes
    ==========
    ego_spawn_idx : int
        Index of spawn location for EV.
    spawn_shift : float or None
        Amount in meters to shift from spawn location along the road.
    n_burn_steps: int
        Number of simulator steps to burn before running motion planner.
    run_steps : int
        Number of simulator steps to run motion planner.
    init_controls : list of util.AttrDict
        Initial control to apply to vehicle
        (i.e. accelerate it to some initial speed, etc).
    controls : util.AttrDict
        The controls to send to PID controller.
    """

    def __init__(self,
            ego_spawn_idx=None,
            spawn_shift=None,
            n_burn_steps=None,
            run_steps=None,
            init_controls=[],
            controls=None
    ):
        self.ego_spawn_idx = ego_spawn_idx
        self.spawn_shift = spawn_shift
        self.n_burn_steps = n_burn_steps
        self.run_steps = run_steps
        self.init_controls = init_controls
        self.controls = controls


def scenario(scenario_params, carla_synchronous, request):
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
        shift = 15
        shift = shift*ego_vehicle.get_transform().get_forward_vector()
        shift = shift + carla.Location(z=30)
        location = ego_vehicle.get_transform().location + shift
        world.get_spectator().set_transform(
            carla.Transform(
                location,
                carla.Rotation(pitch=-90)
            )
        )

        controller = VehiclePIDController(ego_vehicle, max_steering=1.0)
        for idx in range(scenario_params.n_burn_steps):
            for ctrl in scenario_params.init_controls:
                if ctrl.interval[0] <= idx and idx <= ctrl.interval[1]:
                    ego_vehicle.apply_control(ctrl.control)
                    break
            world.tick()
            carlautil.actor_to_speed(ego_vehicle)

        ###############
        # Apply control
        # NOTE: 1 m/s == 3.6 km/h
        speeds = []
        headings = []
        control_steers    = []
        control_throttles = []
        control_brakes    = []
        def add_stats():
            # save the speed, heading and control values
            speed = carlautil.actor_to_speed(ego_vehicle)
            _, heading, _ = carlautil.to_rotation_ndarray(ego_vehicle)
            speeds.append(speed)
            headings.append(heading)
            control = ego_vehicle.get_control()
            control_steers.append(control.steer)
            control_throttles.append(control.throttle)
            control_brakes.append(control.brake)
        add_stats()

        # logging.info(ego_vehicle.get_transform().rotation.yaw)
        controller.set_plan(
            scenario_params.controls.target_speeds,
            scenario_params.controls.target_angles,
            scenario_params.controls.step_period
        )
        for idx in range(scenario_params.run_steps):
            control = controller.step()
            ego_vehicle.apply_control(control)
            world.tick()
            add_stats()
        
        ##########
        # Plotting
        fig, axes = plt.subplots(2,2, figsize=(12, 12))
        axes = axes.ravel()
        timesteps = np.arange(len(speeds)) * 0.05
        axes[0].plot(timesteps, controller.step_to_speed, label="target")
        axes[0].plot(timesteps, speeds, label="measurement")
        axes[0].set_title("Speed response")
        axes[1].plot(timesteps, control_throttles, "b", label="throttle")
        axes[1].plot(timesteps, control_brakes, "r", label="brake")
        axes[1].set_title("Speed input")
        axes[2].plot(timesteps, controller.step_to_angle, label="target")
        _headings = util.npu.warp_radians_about_center(
                np.array(headings), np.array(controller.step_to_angle))
        axes[2].plot(timesteps, _headings, label="measurement")
        axes[2].set_title("Heading response")
        axes[3].plot(timesteps, control_steers, "b", label="steer")
        axes[3].set_title("Steer input")
        for ax in axes:
            # loc = ticker.MultipleLocator(0.5)
            # ax.xaxis.set_major_locator(loc)
            ax.legend()
            ax.set_xlabel("time s")
            ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join("out", f"{request.node.callspec.id}.png"))
        fig.clf()
    
    finally:
        if ego_vehicle:
            ego_vehicle.destroy()

"""
Recall that timesteps are 0.05 s, so 1s has 20 steps.

Out of target speeds 8.33.. m/s, 5.55.. m/s, 2.77.. m/s, the speed
5.55.. m/s causes the most oscillation.

When testing PID controller on speeds 8.33.. m/s, 5.55.. m/s, 2.77.. m/s
the K_P = 0.74, K_D = 0.07, K_I = 0 brings vehicle closest to target
velocity without oscillation, when no steering is involved.

Control input values for PID controller are in the Unreal Engine coordinate system.
"""

enter_rad = math.radians(-179.705383)
CONTROLS_top833 = util.AttrDict(
    target_speeds=[8.33]*12,
    target_angles=[enter_rad]*12,
    step_period=10
)
SCENARIO_top833 = pytest.param(
    # intersection entrance with 4 possible exit lanes.
    # The car enters intersection as angle -179.705383
    # drive straight at 30 km/h == 8.33.. m/s
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=3,
        init_controls=[
            util.AttrDict(
                interval=(0, 2),
                control=carlautil.create_gear_control(throttle=0.01)
            )
        ],
        run_steps=120,
        controls=CONTROLS_top833
    ),
    id="top833"
)

enter_rad = math.radians(-179.705383)
CONTROLS_top555 = util.AttrDict(
    target_speeds=[5.55]*12,
    target_angles=[enter_rad]*12,
    step_period=10
)
SCENARIO_top555 = pytest.param(
    # intersection entrance with 4 possible exit lanes.
    # The car enters intersection as angle -179.705383
    # drive straight at 20 km/h == 5.55.. m/s
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=3,
        init_controls=[
            util.AttrDict(
                interval=(0, 2),
                control=carlautil.create_gear_control(throttle=0.01)
            )
        ],
        run_steps=120,
        controls=CONTROLS_top555
    ),
    id="top555"
)

enter_rad = math.radians(-179.705383)
CONTROLS_top277 = util.AttrDict(
    target_speeds=[2.77]*12,
    target_angles=[enter_rad]*12,
    step_period=10
)
SCENARIO_top277 = pytest.param(
    # intersection entrance with 4 possible exit lanes.
    # The car enters intersection as angle -179.705383
    # drive straight at 10 km/h == 2.77.. m/s
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=3,
        init_controls=[
            util.AttrDict(
                interval=(0, 2),
                control=carlautil.create_gear_control(throttle=0.01)
            )
        ],
        run_steps=120,
        controls=CONTROLS_top277
    ),
    id="top277"
)

enter_rad = math.radians(-179.705383)
CONTROLS_forward = util.AttrDict(
    target_speeds=[5.55]*20,
    target_angles=[enter_rad]*20,
    step_period=10
)
SCENARIO_forward = pytest.param(
    # intersection entrance with 4 possible exit lanes.
    # The car enters intersection as angle -179.705383
    # drive straight at 20 km/h == 5.55 m/s
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=0,
        run_steps=200,
        controls=CONTROLS_forward
    ),
    id="forward"
)

enter_rad = math.radians(-179.705383)
CONTROLS_left052 = util.AttrDict(
    target_speeds=[2.77]*30,
    target_angles=[enter_rad]*10 + [enter_rad - 0.52]*20,
    step_period=10
)
SCENARIO_left052 = pytest.param(
    # Adust heading by 30 degrees == 0.52 radians.
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=3,
        init_controls=[
            util.AttrDict(
                interval=(0, 2),
                control=carlautil.create_gear_control(throttle=0.01)
            )
        ],
        run_steps=300,
        controls=CONTROLS_left052
    ),
    id="left052"
)

enter_rad = math.radians(-179.705383)
CONTROLS_left076 = util.AttrDict(
    target_speeds=[2.77]*30,
    target_angles=[enter_rad]*10 + [enter_rad - 0.76]*20,
    step_period=10
)
SCENARIO_left076 = pytest.param(
    # Adust heading by 45 degrees == 0.785 radians.
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=3,
        init_controls=[
            util.AttrDict(
                interval=(0, 2),
                control=carlautil.create_gear_control(throttle=0.01)
            )
        ],
        run_steps=300,
        controls=CONTROLS_left076
    ),
    id="left076"
)


enter_rad = math.radians(-179.705383)
CONTROLS_left096 = util.AttrDict(
    target_speeds=[2.77]*30,
    target_angles=[enter_rad]*10 + [enter_rad - 0.96]*20,
    step_period=10
)
SCENARIO_left096 = pytest.param(
    # Adust heading by 55 degrees == 0.96 radians.
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=3,
        init_controls=[
            util.AttrDict(
                interval=(0, 2),
                control=carlautil.create_gear_control(throttle=0.01)
            )
        ],
        run_steps=300,
        controls=CONTROLS_left096
    ),
    id="left096"
)

enter_rad = math.radians(-179.705383)
CONTROLS_left_turn035 = util.AttrDict(
    target_speeds=[2.77]*20,
    target_angles=[enter_rad]*10 + [enter_rad - i*0.35 for i in range(0, 10)],
    step_period=10
)
SCENARIO_left_turn035 = pytest.param(
    # While driving at 10km/h, turn 20 deg == 0.35 rad every 0.5 s.
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=10,
        n_burn_steps=10,
        run_steps=200,
        controls=CONTROLS_left_turn035
    ),
    id="left_turn035"
)

enter_rad = math.radians(-179.705383)
CONTROLS_at555_left_turn035 = util.AttrDict(
    target_speeds=[5.55]*20,
    target_angles=[enter_rad]*10 + [enter_rad - i*0.35 for i in range(0, 10)],
    step_period=10
)
SCENARIO_at555_left_turn035 = pytest.param(
    # While driving at 20km/h, turn 20 deg == 0.35 rad every 0.5 s.
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=10,
        run_steps=200,
        controls=CONTROLS_at555_left_turn035
    ),
    id="at555_left_turn035"
)

enter_rad = math.radians(-179.705383)
CONTROLS_at555_right_turn035 = util.AttrDict(
    target_speeds=[5.55]*20,
    target_angles=[enter_rad]*10 + [enter_rad + i*0.35 for i in range(0, 10)],
    step_period=10
)
SCENARIO_at555_right_turn035 = pytest.param(
    # While driving at 20km/h, turn 20 deg == 0.35 rad every 0.5 s.
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=10,
        run_steps=200,
        controls=CONTROLS_at555_right_turn035
    ),
    id="at555_right_turn035"
)

enter_rad = math.radians(89.831024)
CONTROLS_vary_speed1 = util.AttrDict(
    target_speeds=[5.55 - i*(5.55/10) for i in range(0, 6)] \
        + [5.55 - i*(5.55/10) for i in range(4, 0, -1)] \
        + [5.55 - i*(5.55/10) for i in range(0, 6)] \
        + [5.55 - i*(5.55/10) for i in range(4, 0, -1)],
    target_angles=[enter_rad]*20,
    step_period=10
)
SCENARIO_vary_speed1 = pytest.param(
    # Slow down and speed up.
    ScenarioParameters(
        ego_spawn_idx=60,
        spawn_shift=None,
        n_burn_steps=100,
        run_steps=200,
        controls=CONTROLS_vary_speed1,
        init_controls=[
            util.AttrDict(
                interval=(0, 100),
                control=carlautil.create_gear_control(throttle=0.5)
            )
        ],
    ),
    id="vary_speed1"
)

enter_rad = math.radians(-179.705383)
CONTROLS_vary1 = util.AttrDict(
    target_speeds=[5.55 - i*(5.55/10) for i in range(0, 11)] + [5.55 - i*(5.55/10) for i in range(9, 0, -1)],
    target_angles=[enter_rad + i*0.2 for i in range(0, 20)],
    step_period=10
)
SCENARIO_vary1 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=100,
        run_steps=200,
        controls=CONTROLS_vary1,
        init_controls=[
            util.AttrDict(
                interval=(0, 100),
                control=carlautil.create_gear_control(throttle=0.5)
            )
        ],
    ),
    id="vary1"
)

enter_rad = math.radians(-179.705383)
CONTROLS_vary2 = util.AttrDict(
    target_speeds=[5.55 - i*(5.55/10) for i in range(0, 11)] + [5.55 - i*(5.55/10) for i in range(9, 0, -1)],
    target_angles=[enter_rad + i*0.35 for i in range(0, 20)],
    step_period=10
)
SCENARIO_vary2 = pytest.param(
    ScenarioParameters(
        ego_spawn_idx=85,
        spawn_shift=None,
        n_burn_steps=100,
        run_steps=200,
        controls=CONTROLS_vary2,
        init_controls=[
            util.AttrDict(
                interval=(0, 100),
                control=carlautil.create_gear_control(throttle=0.5)
            )
        ],
    ),
    id="vary2"
)

@pytest.mark.parametrize(
    "scenario_params",
    [
        SCENARIO_top833,
        SCENARIO_top555,
        SCENARIO_top277,
        SCENARIO_forward,
        SCENARIO_left052,
        SCENARIO_left076,
        SCENARIO_left096,
        SCENARIO_left_turn035,
        SCENARIO_at555_left_turn035,
        SCENARIO_at555_right_turn035,
        SCENARIO_vary_speed1,
        SCENARIO_vary1,
        SCENARIO_vary2,
    ]
)
def test_Town03_scenario(scenario_params, carla_Town03_synchronous, request):
    scenario(scenario_params, carla_Town03_synchronous, request)
