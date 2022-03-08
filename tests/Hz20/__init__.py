import time

import numpy as np

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
    attach_camera_to_spectator,
    shift_spawn_point
)
from collect.generate import get_all_vehicle_blueprints
from collect.generate import NaiveMapQuerier
from collect.in_simulation.midlevel.v7 import MidlevelAgent
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v3_2.trajectron_scene import (
    TrajectronPlusPlusSceneBuilder
)

DEBUG_SETTINGS = util.AttrDict(
    plot_boundary=False,
    log_agent=False,
    log_cplex=False,
    plot_scenario=True,
    plot_simulation=True,
    plot_overapprox=False,
)

class PlannerScenario(object):

    def __init__(
        self,
        scenario_params: ScenarioParameters,
        ctrl_params: CtrlParameters,
        carla_synchronous: tuple,
        eval_env: Environment,
        eval_stg: Trajectron,
        motion_planner_cls: MidlevelAgent,
        scene_builder_cls: TrajectronPlusPlusSceneBuilder
    ):
        self.client, self.world, self.carla_map, self.traffic_manager = carla_synchronous
        self.scenario_params = scenario_params
        self.ctrl_params = ctrl_params
        self.eval_env = eval_env
        self.eval_stg = eval_stg
        self.motion_planner_cls = motion_planner_cls
        self.scene_builder_cls = scene_builder_cls

    def run(self):
        ego_vehicle = None
        agent = None
        spectator_camera = None
        other_vehicles = []
        record_spectator = False

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
                    vehicle.set_autopilot(True, self.traffic_manager.get_port())
                    if self.scenario_params.ignore_signs:
                        self.traffic_manager.ignore_signs_percentage(vehicle, 100.)
                    if self.scenario_params.ignore_lights:
                        self.traffic_manager.ignore_lights_percentage(vehicle, 100.)
                    if self.scenario_params.ignore_vehicles:
                        self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.)
                    if not self.scenario_params.auto_lane_change:
                        self.traffic_manager.auto_lane_change(vehicle, False)
                    other_vehicles.append(vehicle)
                    other_vehicle_ids.append(vehicle.id)

            frame = self.world.tick()
            agent = self.motion_planner_cls(
                ego_vehicle,
                map_reader,
                other_vehicle_ids,
                self.eval_stg,
                plot_boundary=DEBUG_SETTINGS.plot_boundary,
                log_agent=DEBUG_SETTINGS.log_agent,
                log_cplex=DEBUG_SETTINGS.log_cplex,
                plot_scenario=DEBUG_SETTINGS.plot_scenario,
                plot_simulation=DEBUG_SETTINGS.plot_simulation,
                plot_overapprox=DEBUG_SETTINGS.plot_overapprox,
                scene_builder_cls=self.scene_builder_cls,
                scene_config=online_config,
                **self.scenario_params,
                **self.ctrl_params
            )
            agent.start_sensor()
            assert agent.sensor_is_listening
            
            """Move the spectator to the ego vehicle.
            The positioning is a little off"""
            state = agent.get_vehicle_state()
            goal = agent.get_goal()
            goal_x, goal_y = goal.x, -goal.y
            if goal.is_relative:
                location = carla.Location(
                        x=state[0] + goal_x /2.,
                        y=state[1] - goal_y /2.,
                        z=state[2] + 50)
            else:
                location = carla.Location(
                        x=(state[0] + goal_x) /2.,
                        y=(state[1] + goal_y) /2.,
                        z=state[2] + 50)
            # configure the spectator
            self.world.get_spectator().set_transform(
                carla.Transform(
                    location, carla.Rotation(pitch=-70, yaw=-90, roll=20)
                )
            )
            record_spectator = False
            if record_spectator:
                # attach camera to spectator
                spectator_camera = attach_camera_to_spectator(self.world, frame)

            n_burn_frames = self.scenario_params.n_burn_interval*online_config.record_interval
            if self.ctrl_params.loop_type == LoopEnum.CLOSED_LOOP:
                run_frames = self.scenario_params.run_interval*online_config.record_interval
            else:
                run_frames = self.ctrl_params.control_horizon*online_config.record_interval - 1
            for idx in range(n_burn_frames + run_frames):
                control = None
                for ctrl in self.scenario_params.controls:
                    if ctrl.interval[0] <= idx and idx <= ctrl.interval[1]:
                        control = ctrl.control
                        break
                frame = self.world.tick()
                agent.run_step(frame, control=control)
    
        finally:
            if spectator_camera:
                spectator_camera.destroy()
            if agent:
                agent.destroy()
            if ego_vehicle:
                ego_vehicle.destroy()
            for other_vehicle in other_vehicles:
                other_vehicle.destroy()
        
            if record_spectator == True:
                time.sleep(5)
            else:
                time.sleep(1)
