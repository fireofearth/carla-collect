import os
import time
import math
import logging

import numpy as np
import pandas as pd

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
from collect.in_simulation.midlevel.v8 import MidlevelAgent
from collect.generate.scene import OnlineConfig
from collect.generate.scene.v3_2.trajectron_scene import TrajectronPlusPlusSceneBuilder
from collect.exception import InSimulationException


class MonteCarloScenario(object):

    DEBUG_SETTINGS = util.AttrDict(
        plot_boundary=False,
        log_agent=False,
        log_cplex=False,
        plot_scenario=False,
        plot_simulation=False,
        plot_overapprox=False,
    )

    TOL = 5

    def __init__(
        self,
        scenario_params: ScenarioParameters,
        ctrl_params: CtrlParameters,
        carla_synchronous: tuple,
        eval_env: Environment,
        eval_stg: Trajectron,
        motion_planner_cls: MidlevelAgent,
        scene_builder_cls: TrajectronPlusPlusSceneBuilder,
        n_simulations=2
    ):
        self.client, self.world, self.carla_map, self.traffic_manager = carla_synchronous
        self.scenario_params = scenario_params
        self.ctrl_params = ctrl_params
        self.eval_env = eval_env
        self.eval_stg = eval_stg
        self.motion_planner_cls = motion_planner_cls
        self.scene_builder_cls = scene_builder_cls
        self.n_simulations = n_simulations

        self.map_reader = NaiveMapQuerier(self.world, self.carla_map, debug=True)
        self.online_config = OnlineConfig(record_interval=10, node_type=self.eval_env.NodeType)
        
        # Mock vehicles
        self.spawn_points = self.carla_map.get_spawn_points()
        self.blueprints = get_all_vehicle_blueprints(self.world)
        self.blueprint_audi_a2 = self.world.get_blueprint_library().find('vehicle.audi.a2')

    def episode(self, episode_idx):
        logging.info(f"doing episode {episode_idx}")
        ego_vehicle = None
        agent = None
        other_vehicles = []
        stats = util.AttrDict(
            success=False,
            infeasibility=False,
            steps=0,
            plan_steps=0
        )

        try:
            spawn_indices = [self.scenario_params.ego_spawn_idx] + self.scenario_params.other_spawn_ids
            other_vehicle_ids = []
            for k, spawn_idx in enumerate(spawn_indices):
                if k == 0:
                    blueprint = self.blueprint_audi_a2
                else:
                    blueprint = np.random.choice(self.blueprints)
                spawn_point = self.spawn_points[spawn_idx]
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
                self.map_reader,
                other_vehicle_ids,
                self.eval_stg,
                scene_builder_cls=self.scene_builder_cls,
                scene_config=self.online_config,
                **self.scenario_params,
                **self.ctrl_params,
                **self.DEBUG_SETTINGS,
            )
            agent.start_sensor()
            assert agent.sensor_is_listening
            if self.scenario_params.goal:
                agent.set_goal(**self.scenario_params.goal)

            """Setup vehicle routes"""
            if "CARLANAME" in os.environ and os.environ["CARLANAME"] == "carla-0.9.13":
                for k, vehicle in enumerate(other_vehicles):
                    route = None
                    try:
                        route = self.scenario_params.other_routes[k]
                        len(route)
                    except (TypeError, IndexError) as e:
                        continue
                    self.traffic_manager.set_route(vehicle, route)
            else:
                logging.info("Skipping setting up OV routes.")
            
            if episode_idx == 0:
                """Move the spectator to the ego vehicle.
                The positioning is a little off"""
                goal = agent.get_goal()
                goal_x, goal_y = goal.x, -goal.y
                state = agent.get_vehicle_state()
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
                # rotation = carla.Rotation(pitch=-70, yaw=-90, roll=20)
                rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)
                # configure the spectator
                self.world.get_spectator().set_transform(
                    carla.Transform(location, rotation)
                )
                location = carla.Location(goal_x, goal_y, state[2])
                carlautil.debug_point(
                    self.world, location, t=60.0, label="goal"
                )

            n_burn_frames = self.scenario_params.n_burn_interval*self.online_config.record_interval
            if self.ctrl_params.loop_type == LoopEnum.CLOSED_LOOP:
                run_frames = self.scenario_params.run_interval*self.online_config.record_interval
            else:
                run_frames = self.ctrl_params.control_horizon*self.online_config.record_interval - 1
            for idx in range(n_burn_frames):
                control = None
                for ctrl in self.scenario_params.controls:
                    if ctrl.interval[0] <= idx and idx <= ctrl.interval[1]:
                        control = ctrl.control
                        break
                agent.run_step(frame, control=control)
                frame = self.world.tick()

            for idx in range(run_frames):
                agent.run_step(frame)
                frame = self.world.tick()
                stats.steps += 1
                state = agent.get_vehicle_state(flip_y=True)
                goal = agent.get_goal()
                dist = math.sqrt((state[0] - goal.x)**2 + (state[1] - goal.y)**2)
                if dist < self.TOL:
                    stats.success = True
                    break
        
        except InSimulationException as e:
            stats.infeasibility = True

        finally:
            if agent:
                agent.destroy()
            if ego_vehicle:
                ego_vehicle.destroy()
            for other_vehicle in other_vehicles:
                other_vehicle.destroy()
            stats.plan_steps = stats.steps / self.online_config.record_interval
            time.sleep(1)
        
        logging.info(
            f"episode succeeded? {stats.success}. "
            f"Infeasibility? {stats.infeasibility}. "
            f"Ran planner over {stats.steps} total steps; "
            f"{stats.plan_steps} planning steps."
        )
        return pd.Series(stats)

    def run(self):
        logging.info(f"Running {self.n_simulations} episodes.")
        stats = []
        for episode_idx in range(self.n_simulations):
            stats.append(self.episode(episode_idx))
        stats = pd.DataFrame(stats)
        frac_success = stats["success"].mean()
        frac_infeasi = stats["infeasibility"].mean()
        mean_steps   = stats[stats["success"]]["steps"].mean()
        mean_plan    = stats[stats["success"]]["plan_steps"].mean()
        logging.info(f"frac of success: {frac_success}")
        logging.info(f"frac of infeasibility: {frac_infeasi}")
        logging.info(f"success mean steps: {mean_steps}")
        logging.info(f"success mean planning steps: {mean_plan}")


class PlannerScenario(object):

    DEBUG_SETTINGS = util.AttrDict(
        plot_boundary=False,
        log_agent=False,
        log_cplex=False,
        plot_scenario=True,
        plot_simulation=True,
        # plot_scenario=False,
        # plot_simulation=False,
        plot_overapprox=False,
    )

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
                scene_builder_cls=self.scene_builder_cls,
                scene_config=online_config,
                **self.scenario_params,
                **self.ctrl_params,
                **self.DEBUG_SETTINGS,
            )
            agent.start_sensor()
            assert agent.sensor_is_listening

            """Setup vehicle routes"""
            if "CARLANAME" in os.environ and os.environ["CARLANAME"] == "carla-0.9.13":
                for k, vehicle in enumerate(other_vehicles):
                    route = None
                    try:
                        route = self.scenario_params.other_routes[k]
                        len(route)
                    except (TypeError, IndexError) as e:
                        continue
                    self.traffic_manager.set_route(vehicle, route)
            else:
                logging.info("Skipping setting up OV routes.")
            
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
