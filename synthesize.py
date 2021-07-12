#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Generate CARLA batched data for Trajectron++

To test, call
python synthesize.py -e1 -f200 -b10
"""

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import time
import numpy as np
import dill

import carla
from carla import VehicleLightState as vls
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
SetVehicleLightState = carla.command.SetVehicleLightState
FutureActor = carla.command.FutureActor

try:
    # trajectron-plus-plus/trajectron
    from environment import Environment, Scene, Node
    from environment import GeometricMap, derivative_of
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from collect.generate import (
        get_all_vehicle_blueprints,
        DataCollector, IntersectionReader, SampleLabelFilter,
        ScenarioIntersectionLabel, ScenarioSlopeLabel)
from collect.generate import SceneConfig
from collect.generate.scene.v2_1.trajectron_scene import TrajectronPlusPlusSceneBuilder
from collect.generate.scene.v2_1.trajectron_scene import (
        print_and_reset_specs)
from collect.generate.scene.trajectron_util import (
        standardization, plot_trajectron_scene)

class DataGenerator(object):

    def __init__(self, args):
        self.args = args
        # n_episdoes : int
        #     Number of episodes to collect data.
        self.n_episodes = self.args.n_episodes
        # n_frames : int
        #     Number of frames to collect data in each episode.
        #     Note: a vehicle takes roughly 130 frames to make a turn.
        self.n_frames = self.args.n_frames
        # delta : float
        #     Step size for synchronous mode.
        self.delta = 0.1
        if self.args.seed is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(self.args.seed)

        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)
        if self.args.map is None:
            self.world = self.client.get_world()
            logging.info("Using the current map.")
        else:
            logging.info(f"Using the map {self.args.map}.")
            self.world = self.client.load_world(self.args.map)
        self.carla_map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.intersection_reader = IntersectionReader(
                self.world, self.carla_map, debug=self.args.debug)
        
        self.env = Environment(node_type_list=['VEHICLE'],
                standardization=standardization)
        attention_radius = dict()
        attention_radius[(self.env.NodeType.VEHICLE, self.env.NodeType.VEHICLE)] = 30.0
        self.env.attention_radius = attention_radius
        self.env.robot_type = self.env.NodeType.VEHICLE
        self.scenes = []

        self.scene_config = SceneConfig(
                scene_interval=self.args.scene_length,
                node_type=self.env.NodeType)

        # exclude_filter : SampleLabelFilter
        #     Filter for slopes and controlled intersections
        self.exclude_filter = SampleLabelFilter(
            # intersection_type=[ScenarioIntersectionLabel.CONTROLLED],
            slope_type=[ScenarioSlopeLabel.SLOPES]
        )

    def add_scene(self, scene):
        self.scenes.append(scene)

    def __setup_actors(self, episode):
        """Setup vehicles and data collectors for an episode.

        Parameters
        ----------
        episode : int

        Returns
        -------
        list of int
            IDs of 4 wheeled vehicles on autopilot.
        list of DataCollector
            Data collectors with set up done and listening to LIDAR.
        """
        vehicle_ids = []
        data_collectors = []
        blueprints = get_all_vehicle_blueprints(self.world)
        spawn_points = self.carla_map.get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        logging.info(f"Using {self.args.n_vehicles} out of "
                f"{number_of_spawn_points} spawn points")

        if self.args.n_vehicles < number_of_spawn_points:
            np.random.shuffle(spawn_points)
        elif self.args.n_vehicles > number_of_spawn_points:
            msg = "requested %d vehicles, but could only find %d spawn points"
            logging.warning(msg, self.args.n_vehicles, number_of_spawn_points)
            self.args.n_vehicles = number_of_spawn_points

        # Generate vehicles
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self.args.n_vehicles:
                break
            blueprint = np.random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if self.args.car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        # Wait for vehicles to finish generating
        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                vehicle_ids.append(response.actor_id)
        
        # Add data collector to a handful of vehicles
        vehicles = self.world.get_actors(vehicle_ids)
        vehicles = dict(zip(vehicle_ids, vehicles))
        vehicles_ids_to_data_collect = vehicle_ids[:self.args.n_data_collectors]

        
        for idx, vehicle_id in enumerate(vehicles_ids_to_data_collect):
            vehicle_ids_to_watch = vehicle_ids[:idx] + vehicle_ids[idx + 1:]
            vehicle = self.world.get_actor(vehicle_id)
            data_collector = DataCollector(vehicle,
                    self.intersection_reader,
                    vehicle_ids_to_watch,
                    scene_builder_cls=TrajectronPlusPlusSceneBuilder,
                    scene_config=self.scene_config,
                    save_frequency=self.args.save_frequency,
                    n_burn_frames=self.args.n_burn_frames,
                    episode=episode,
                    exclude_samples=self.exclude_filter,
                    callback=self.add_scene,
                    debug=self.args.debug)
            data_collector.start_sensor()
            data_collectors.append(data_collector)

        logging.info(f"Spawned {len(vehicle_ids)} vehicles")
        logging.info(f"Spawned {len(data_collectors)} data collectors")
        return vehicle_ids, data_collectors
    
    def __run_episode(self, episode):
        """Run a dataset collection episode.
        
        Parameters
        ----------
        episode : int
            The index of the episode to run.
        """
        vehicle_ids = []
        data_collectors = []

        try:
            logging.info("Create vehicles and data collectors.")
            vehicle_ids, data_collectors = self.__setup_actors(episode)
            logging.info("Running simulation.")
            for idx in range(self.n_frames):
                frame = self.world.tick()
                for data_collector in data_collectors:
                    data_collector.capture_step(frame)

        finally:
            logging.info(f"Ending episode {episode}.")
            logging.info("Destroying data collectors.")
            if data_collectors:
                for data_collector in data_collectors:
                    data_collector.destroy()
            
            logging.info("Destroying vehicles.")
            if vehicle_ids:
                self.client.apply_batch(
                        [carla.command.DestroyActor(x) for x in vehicle_ids])

    def run(self):
        """Main entry point to data collection."""
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
            if self.args.seed is not None:
                self.traffic_manager.set_random_device_seed(self.args.seed)        
            if self.args.hybrid:
                self.traffic_manager.set_hybrid_physics_mode(True)
            self.world.apply_settings(settings)
            if self.args.debug:
                self.intersection_reader.debug_display()

            for episode in range(self.args.start_at_episode,
                    self.args.start_at_episode + self.n_episodes):
                logging.info(f"Running episode {episode}.")
                self.__run_episode(episode)

        finally:
            logging.info("Reverting to original settings.")
            if original_settings:
                self.world.apply_settings(original_settings)
        
        print(f"Generated { len(self.scenes) } scenes.")
        print_and_reset_specs()

        logging.info("Plotting some scenes.")
        if len(self.scenes) > 6:
            scenes_to_plot = random.choices(self.scenes, k=6)
        else:
            scenes_to_plot = self.scenes
        for scene in scenes_to_plot:
            plot_trajectron_scene(self.args.save_directory, scene)
        
        if len(self.scenes) == 0:
            logging.info("No scenes collected. Closing.")
            return

        self.env.scenes = self.scenes
        os.makedirs(self.args.save_directory, exist_ok=True)
        filename = "{}_{}.pkl".format(
                datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S"),
                self.args.save_label)
        savepath = os.path.join(self.args.save_directory, filename)
        logging.info(f"Saving dataset as {savepath}")
        with open(savepath, 'wb') as f:
            dill.dump(self.env, f, protocol=dill.HIGHEST_PROTOCOL)
        logging.info("Finished run.")


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

def dir_path(s):
    if os.path.isdir(s):
        return s
    else:
        raise argparse.ArgumentTypeError(
                f"readable_dir:{s} is not a valid path")

def main():
    """Main method"""
    
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Show debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--dir',
        type=dir_path,
        default='out',
        dest='save_directory',
        help='Directory to save the dataset (default: out).')
    argparser.add_argument(
        '--label',
        type=str,
        default='dataset',
        dest='save_label',
        help='Label of dataset.')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--map',
        type=str,
        help="Set the CARLA map to collect data from.")
    argparser.add_argument(
        '-e', '--n-episodes',
        metavar='E',
        default=5,
        type=int,
        help='Number of episodes to run (default: 5)')
    argparser.add_argument(
        '--start-at-episode',
        metavar='S',
        default=0,
        type=int,
        help='the episode number to start indexing at (default: 0)')
    argparser.add_argument(
        '-f', '--n-frames',
        metavar='F',
        default=500,
        type=int,
        help='Number of frames in each episode to capture (default: 500)')
    argparser.add_argument(
        '-b', '--n-burn-frames',
        metavar='B',
        default=60,
        type=int,
        help="Number of frames at the beginning of each episode to skip data collection (default: 60)")
    argparser.add_argument(
        '-n', '--n-vehicles',
        metavar='N',
        default=80,
        type=int,
        help='number of vehicles (default: 80)')
    argparser.add_argument(
        '-d', '--n-data-collectors',
        metavar='D',
        default=20,
        type=int,
        help='number of data collectos to add on vehicles (default: 20)')
    argparser.add_argument(
        '--save-frequency',
        metavar='S',
        default=26,
        type=int)
    argparser.add_argument(
        '--scene-length',
        metavar='L',
        default=25,
        type=int)
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        help='Enable car lights')

    args = argparser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)

    try:
        generator = DataGenerator(args)
        generator.run()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
