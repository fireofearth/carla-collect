#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

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

import carla
from carla import VehicleLightState as vls
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
SetVehicleLightState = carla.command.SetVehicleLightState
FutureActor = carla.command.FutureActor

from generate.data import (
        DataCollector, Map10HDBoundTIntersectionReader, SampleLabelFilter,
        ScenarioIntersectionLabel, ScenarioSlopeLabel, BoundingRegionLabel)

class DataGenerator(object):

    def __init__(self, args):
        self.args = args
        # n_episdoes : int
        #     Number of episodes to collect data.
        self.n_episodes = self.args.n_episodes
        # n_frames : int
        #     Number of frames to collect data in each episode.
        #     Note: a vehicle takes roughly 130 frames to make a turn.
        self.n_frames = 200
        # n_burn_frames : int
        #     The number of initial frames to skip before saving data.
        self.n_burn_frames = 0
        # save_frequency : int
        #     Size of interval in frames to wait between saving.
        self.save_frequency = 5
        # delta : float
        #     Step size for synchronous mode.
        #     Note: setting step size to t=0.2 corresponds to frequence 5HZ
        #     And 4 seconds of future positions when using ESP parameter T=20
        self.delta = 0.2
        if self.args.seed is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(self.args.seed)

        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        if self.carla_map.name != "Town10HD":
            self.world = self.client.load_world("Town10HD")
            self.carla_map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(8000)        
        self.intersection_reader = Map10HDBoundTIntersectionReader(
                self.world, self.carla_map, debug=self.args.debug)
        
        # filtering out controlled intersections
        self.exclude_filter = SampleLabelFilter()

    def __setup_vehicle(self, blueprint, spawn_point_indices, spawn_points):
        if blueprint.has_attribute('color'):
            color = np.random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')
        transform = spawn_points[random.choice(spawn_point_indices)]
        return self.world.spawn_actor(blueprint, transform)

    def __setup_actors(self, episode):
        """Setup vehicles and data collectors for an episode.
        INVARIANT: CARLA world has already been set to synchronous mode.

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
        ego_vehicle_blueprint = self.world.get_blueprint_library().find("vehicle.audi.a2")
        other_vehicle_blueprint = self.world.get_blueprint_library().find("vehicle.audi.etron")

        spawn_points = self.carla_map.get_spawn_points()
        # spawn point indices sorted from closest to intersection 
        ego_vehicle_spawn_point_idx = [30, 29, 28]
        other_vehicle_spawn_point_idx = [36, 35]
        # specify spawn points
        # ego_vehicle_spawn_point_idx = [152]
        # other_vehicle_spawn_point_idx = [28]

        """Set the vehicles."""
        ego_vehicle = self.__setup_vehicle(
                ego_vehicle_blueprint,
                ego_vehicle_spawn_point_idx,
                spawn_points)
        other_vehicle = self.__setup_vehicle(
                other_vehicle_blueprint,
                other_vehicle_spawn_point_idx,
                spawn_points)
        ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        other_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self.world.tick()
        vehicle_ids = [ego_vehicle.id, other_vehicle.id]

        """Set the data collector."""
        data_collector = DataCollector(ego_vehicle,
                self.intersection_reader,
                save_frequency=self.save_frequency,
                save_directory=self.args.save_directory,
                n_burn_frames=self.n_burn_frames,
                episode=episode,
                exclude_samples=self.exclude_filter,
                should_augment=self.args.augment,
                n_augments=self.args.n_augments,
                debug=self.args.debug)
        data_collector.start_sensor()
        data_collector.set_vehicles([other_vehicle.id])
        data_collectors = [data_collector]

        logging.info(f"spawned {len(vehicle_ids)} vehicles")
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
            settings.max_substep_delta_time = self.max_substep_delta_time
            settings.max_substeps = self.max_substeps
            settings.fixed_delta_seconds = self.delta
            settings.synchronous_mode = True

            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
            self.traffic_manager.global_percentage_speed_difference(0.0)
            if self.args.seed is None:
                self.traffic_manager.set_random_device_seed(
                        random.randint(0, 99999))
            else:
                self.traffic_manager.set_random_device_seed(self.args.seed)
            if self.args.hybrid:
                self.traffic_manager.set_hybrid_physics_mode(True)
            self.world.apply_settings(settings)
            if self.args.debug:
                self.intersection_reader.debug_display()

            for episode in range(self.n_episodes):
                logging.info(f"Running episode {episode}.")
                self.__run_episode(episode)

        finally:
            logging.info("Reverting to original settings.")
            if original_settings:
                self.world.apply_settings(original_settings)


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
        help='Directory to save the samples.')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '-e', '--n-episodes',
        metavar='E',
        default=20,
        type=int,
        help='Number of episodes to run (default: 10)')
    argparser.add_argument(
        '--augment-data',
        action='store_true',
        dest='augment',
        help='Enable data augmentation')
    argparser.add_argument(
        '--n-augments',
        default=1,
        type=int,
        help=("Number of augmentations to create from each sample. "
        "If --n-aguments=5 then a random number from 1 to 5 augmentations will be produced from each sample"))
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')

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
