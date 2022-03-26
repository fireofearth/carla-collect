"""In-simulation control method."""

import sys
import os
import json
import math
import logging
import collections
import datetime
import dill

import numpy as np
import torch
# import pygame
# from pygame.locals import *
import carla

import utility as util
import carlautil
import carlautil.debug

try:
    # imports from trajectron-plus-plus/trajectron
    from environment import Environment, Scene
    from model.model_registrar import ModelRegistrar
    from model import Trajectron
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from .midlevel.v1 import LCSSHighLevelAgent
from ..generate import get_all_vehicle_blueprints
from ..generate.map import NaiveMapQuerier
from ..generate.scene import OnlineConfig
from ..generate.scene.v2_1.trajectron_scene import (
        standardization, print_and_reset_specs, plot_trajectron_scene)

def load_model(model_dir, env, ts=3999, device='cpu'):
    model_registrar = ModelRegistrar(model_dir, device)
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    hyperparams['map_enc_dropout'] = 0.0
    if 'incl_robot_node' not in hyperparams:
        hyperparams['incl_robot_node'] = False

    stg = Trajectron(model_registrar, hyperparams,  None, device)
    stg.set_environment(env)
    stg.set_annealing_params()
    return stg, hyperparams

def get_approx_union(theta, vertices):
    """Gets A_t, b_0 for the contraint set A_t x >= b_0
    vertices : np.array
        Vertices of shape (?, 8)
    
    Returns
    =======
    np.array
        A_t matrix of shape (4, 2)
    np.array
        b_0 vector of shape (4,)
    """
    At = np.array([
            [ np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])
    At = np.concatenate((np.eye(2), -np.eye(2),)) @ At

    a0 = np.max(At @ vertices[:, 0:2].T, axis=1)
    a1 = np.max(At @ vertices[:, 2:4].T, axis=1)
    a2 = np.max(At @ vertices[:, 4:6].T, axis=1)
    a3 = np.max(At @ vertices[:, 6:8].T, axis=1)
    b0 = np.max(np.stack((a0, a1, a2, a3)), axis=0)
    return At, b0

class OnlineManager(object):
    """
    TODO: unused
    TODO: some values are hardcoded.
    """
    
    def __init__(self, args):
        self.args = args
        self.delta = 0.1
        
        """Load dummy dataset."""
        eval_scene = Scene(timesteps=25, dt=0.5, name='test')
        eval_env = Environment(node_type_list=['VEHICLE'],
                standardization=standardization)
        attention_radius = dict()
        attention_radius[(eval_env.NodeType.VEHICLE, eval_env.NodeType.VEHICLE)] = 30.0
        eval_env.attention_radius = attention_radius
        eval_env.robot_type = eval_env.NodeType.VEHICLE
        eval_env.scenes = [eval_scene]

        """Load model."""
        model_dir = 'experiments/nuScenes/models/20210622'
        model_name = 'models_19_Mar_2021_22_14_19_int_ee_me_ph8'
        model_path = os.path.join(os.environ['TRAJECTRONPP_DIR'],
                model_dir, model_name)
        self.eval_stg, self.stg_hyp = load_model(
                model_path, eval_env, ts=20)#, device='cuda:0')

        """Get CARLA connectors"""
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.map_reader = NaiveMapQuerier(
                self.world, self.carla_map, debug=self.args.debug)
        self.online_config = OnlineConfig(node_type=eval_env.NodeType)

    
    def create_agents(self, params):
        spawn_points = self.carla_map.get_spawn_points()
        spawn_point = spawn_points[params.ego_spawn_idx]
        blueprint = self.world.get_blueprint_library().find('vehicle.audi.a2')
        params.ego_vehicle = self.world.spawn_actor(blueprint, spawn_point)

        other_vehicle_ids = []
        blueprints = get_all_vehicle_blueprints(self.world)
        for idx in params.other_spawn_ids:
            blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[idx]
            other_vehicle = self.world.spawn_actor(blueprint, spawn_point)
            other_vehicle.set_autopilot(True, self.traffic_manager.get_port())
            params.other_vehicles.append(other_vehicle)
            other_vehicle_ids.append(other_vehicle.id)

        """Get our high level APIs"""
        params.agent = LCSSHighLevelAgent(
                params.ego_vehicle,
                self.map_reader,
                other_vehicle_ids,
                self.eval_stg,
                scene_config=self.online_config)
        params.agent.start_sensor()
        return params

    def test_scenario(self):
        params = util.AttrDict(ego_spawn_idx=2, other_spawn_ids=[104, 102, 235],
                ego_vehicle=None, other_vehicles=[], agent=None)
        n_burn_frames = 3*self.online_config.record_interval
        try:
            params = self.create_agents(params)
            agent = params.agent

            for idx in range(n_burn_frames \
                    + 30*self.online_config.record_interval):
                frame = self.world.tick()
                agent.run_step(frame)

        finally:
            if params.agent:
                params.agent.destroy()
            if params.ego_vehicle:
                params.ego_vehicle.destroy()
            for other_vehicle in params.other_vehicles:
                other_vehicle.destroy()

    def run(self):
        """Main entry point to simulatons."""
        original_settings = None

        try:
            logging.info("Enabling synchronous setting and updating traffic manager.")
            original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = self.delta
            settings.synchronous_mode = True

            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
            self.traffic_manager.global_percentage_speed_difference(0.0)
            self.world.apply_settings(settings)
            self.test_scenario()

        finally:
            logging.info("Reverting to original settings.")
            if original_settings:
                self.world.apply_settings(original_settings)
