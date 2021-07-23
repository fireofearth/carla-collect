import os
import collections
from abc import ABC, abstractmethod
import copy
import logging

import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import carla

import utility as util
import carlautil
import carlautil.debug

from ..node import NodeTypeEnum
from ..map import MapQuerier
from ..label import SampleLabelFilter
from ..label import SegmentationLabel

class AbstractSceneConfig(ABC):
    """Scene configuration. Scene builder and classes inheriting it
    uses this information to get parameters for scene processing.
    """
    def __init__(self,
            record_interval=None,
            pixels_per_m=None,
            node_type=None):
        self.record_interval = record_interval
        self.pixels_per_m = pixels_per_m
        self.node_type = node_type


class SceneConfig(AbstractSceneConfig):
    """Self contained scene configuration.
    
    Attributes
    ==========
    scene_interval : int
        The length of the scene.
        The scene builder will call SceneBuilder.finish_scene()
        after collecting scene_interval trajectory points.
    record_interval : int
        The interval of time between capturing trajectories.
        If simulator stimestep is 0.1 and record_interval is 5,
        then trajectories in the scene are sampled at 0.5s or 5 Hz.
    """
    def __init__(self,
            scene_interval=32,
            record_interval=5,
            pixels_per_m=3,
            node_type=NodeTypeEnum(['VEHICLE'])):
        super().__init__(
                record_interval=record_interval,
                pixels_per_m=pixels_per_m,
                node_type=node_type)
        self.scene_interval = scene_interval


class OnlineConfig(AbstractSceneConfig):
    """
    Scene builder will *not* call SceneBuilder.finish_scene(). The user
    must call SceneBuilder.finish_scene() or SceneBuilder.get_scene() manually.
    """
    def __init__(self,
            record_interval=5,
            pixels_per_m=3,
            node_type=NodeTypeEnum(['VEHICLE'])):
        super().__init__(
                record_interval=record_interval,
                pixels_per_m=pixels_per_m,
                node_type=node_type)


class SceneBuilderData(object):
    def __init__(self, save_directory, scene_name, map_name, fixed_delta_seconds,
            trajectory_data, overhead_points, overhead_labels, overhead_ids,
            map_data, vehicle_visibility,
            sensor_loc_at_t0, first_frame, scene_config):
        self.save_directory = save_directory
        self.scene_name = scene_name
        self.map_name = map_name
        self.fixed_delta_seconds = fixed_delta_seconds
        self.trajectory_data = trajectory_data
        self.overhead_points = overhead_points
        self.overhead_labels = overhead_labels
        self.overhead_ids = overhead_ids
        self.map_data = map_data
        self.vehicle_visibility = vehicle_visibility
        self.sensor_loc_at_t0 = sensor_loc_at_t0
        self.first_frame = first_frame
        self.scene_config = scene_config


class SceneBuilder(ABC):
    """Constructs a scene that contains samples
    of consecutive snapshots of the simulation. The scenes have:

    - Vehicle bounding boxes and locations in cartesion coordinates.
    - LIDAR overhead bitmap consisting of LIDAR points collected at
    each timestep of the ego vehicle.
    
    There are two models:
    
    - Static scene mode: builder collects data over a scene_interval
    length ego vehicle trajectory and calls its own method
    finish_scene() when its done.
    - Online scene mode: builder continually builds the scene and
    returns whatever it has collected using get_scene()."""

    Z_UPPERBOUND =  4
    Z_LOWERBOUND = -4
    
    def __init__(self, data_collector,
            map_reader,
            ego_vehicle,
            other_vehicles,
            lidar_feeds,
            scene_name,
            first_frame,
            scene_config=SceneConfig(),
            save_directory='out',
            exclude_samples=SampleLabelFilter(),
            scene_radius=70.0,
            callback=lambda x: x,
            debug=False):
        """
        pixel_dim : np.array
            Dimension of a pixel in meters (m)
        """
        # __data_collector : DataCollector
        self.__data_collector = data_collector
        # __map_reader : MapQuerier
        self.__map_reader = map_reader
        # __ego_vehicle : carla.Vehicle
        self.__ego_vehicle = ego_vehicle
        # __other_vehicles : list of carla.Vehicle
        self.__other_vehicles = other_vehicles
        # __lidar_feeds : collections.OrderedDict
        self.__lidar_feeds = lidar_feeds
        # __scene_name : str
        self.__scene_name = scene_name
        # __first_frame : int
        self.__first_frame = first_frame
        # __world : carla.World
        self.__world = self.__ego_vehicle.get_world()
        self.__save_directory = save_directory
        # __exclude_samples :  SampleLabelFilter
        self.__exclude_samples = exclude_samples
        # __callback : function
        self.__callback = callback
        self.__debug = debug

        # __scene_config : AbstractSceneConfig
        #   If it is of type SceneConfig then scene builder
        #   collecting scene once.
        #   If it is of type OnlineConfig then scene builder
        #   can be reused to collect multipe scenes.
        self.__scene_config = scene_config
        if self.is_online:
            self.__last_frame = np.inf
        else:
            self.__last_frame = self.__first_frame + (self.__scene_config.scene_interval \
                    - 1)*self.__scene_config.record_interval + 1

        self.__finished_lidar_trajectory = False
        
        # __seen_lidar_keys : set
        #   Used by online builder to keep track of frames
        #   where it collected LIDAR from.
        self.__seen_lidar_keys = set()

        # __vehicle_visibility : dict of (int, set of int)
        #   IDs of vehicles visible to the ego vehicle by frame ID.
        #   Vehicles are visible when then are hit by semantic LIDAR.
        self.__vehicle_visibility = dict()

        self.__radius = scene_radius
        # __sensor_loc_at_t0 : carla.Transform
        self.__sensor_loc_at_t0 = None
        # __overhead_points : np.float32
        self.__overhead_points = None
        # __overhead_labels : np.uint32
        self.__overhead_labels = None
        # __overhead_ids : np.uint32
        self.__overhead_ids = None
        # __trajectory_data : pd.DataFrame
        self.__trajectory_data = None
    
    @property
    def is_online(self):
        return isinstance(self.__scene_config, OnlineConfig)

    def debug_draw_red_player_bbox(self):
        self.__world.debug.draw_box(
                carla.BoundingBox(
                    self.__ego_vehicle.get_transform().location,
                    self.__ego_vehicle.bounding_box.extent),
                self.__ego_vehicle.get_transform().rotation,
                thickness=0.5,
                color=carla.Color(r=255, g=0, b=0, a=255),
                life_time=3.0)

    def debug_draw_green_player_bbox(self):
        self.__world.debug.draw_box(
                carla.BoundingBox(
                        self.__ego_vehicle.get_transform().location,
                        self.__ego_vehicle.bounding_box.extent),
                self.__ego_vehicle.get_transform().rotation,
                thickness=0.5,
                color=carla.Color(r=0, g=255, b=0, a=255),
                life_time=3.0)

    def __should_exclude_dataset_sample(self, labels):
        for key, val in vars(labels).items():
            if self.__exclude_samples.contains(key, val):
                if self.__debug:
                    self.debug_draw_red_player_bbox()
                    logging.debug("filter dataset sample")
                return True
        
        if self.__debug:
            self.debug_draw_green_player_bbox()
        return False
    
    def __capture_agents_within_radius(self, frame_id):
        labels = self.__map_reader.get_actor_labels(self.__ego_vehicle)
        if self.__should_exclude_dataset_sample(labels):
            self.__remove_scene_builder()
        player_location = carlautil.actor_to_location_ndarray(self.__ego_vehicle)
        if len(self.__other_vehicles):
            other_ids = list(self.__other_vehicles.keys())
            other_vehicles = self.__other_vehicles.values()
            others_data = carlautil.actors_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(other_vehicles).T
            other_locations = others_data[:3].T
            distances = np.linalg.norm(other_locations - player_location, axis=1)
            df = pd.DataFrame({
                    'frame_id': np.full((len(other_ids),), frame_id),
                    'type': [self.__scene_config.node_type.VEHICLE] * len(other_ids),
                    'node_id': util.map_to_list(str, other_ids),
                    'robot': [False] * len(other_ids),
                    'distances': distances,
                    'x': others_data[0],
                    'y': others_data[1],
                    'z': others_data[2],
                    'v_x': others_data[3],
                    'v_y': others_data[4],
                    'v_z': others_data[5],
                    'a_x': others_data[6],
                    'a_y': others_data[7],
                    'a_z': others_data[8],
                    'length': others_data[9],
                    'width': others_data[10],
                    'height': others_data[11],
                    'heading': others_data[13]})
            df = df[df['distances'] < self.__radius]
            df = df[df['z'].between(self.Z_LOWERBOUND, self.Z_UPPERBOUND, inclusive=False)]
            del df['distances']
        else:
            df = pd.DataFrame(columns=['frame_id', 'type', 'node_id', 'robot',
                    'x', 'y', 'z', 'length', 'width', 'height', 'heading'])
        ego_data = carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(self.__ego_vehicle)
        data_point = pd.Series({
                'frame_id': frame_id,
                'type': self.__scene_config.node_type.VEHICLE,
                'node_id': 'ego',
                'robot': True,
                'x': ego_data[0],
                'y': ego_data[1],
                'z': ego_data[2],
                'v_x': ego_data[3],
                'v_y': ego_data[4],
                'v_z': ego_data[5],
                'a_x': ego_data[6],
                'a_y': ego_data[7],
                'a_z': ego_data[8],
                'length': ego_data[9],
                'width': ego_data[10],
                'height': ego_data[11],
                'heading': ego_data[13]})
        df = df.append(data_point, ignore_index=True)
        if self.__trajectory_data is None:
            self.__trajectory_data = df
        else:
            self.__trajectory_data = pd.concat((self.__trajectory_data, df),
                    ignore_index=True)

    def __lidar_snapshot_to_populate_vehicle_visibility(self, lidar_measurement,
            points, labels, object_ids):
        """Called by capture_lidar() to obtain vehicle visibility from
        segmented LIDAR points.
        """
        frame = lidar_measurement.frame
        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
            vehicle_label_mask = labels == SegmentationLabel.Vehicle.value
            object_ids = np.unique(object_ids[vehicle_label_mask])
            self.__vehicle_visibility[frame_id] = set(util.map_to_list(str, object_ids))
            self.__vehicle_visibility[frame_id].add('ego')

    def __add_1_to_points(self, points):
        return np.pad(points, [(0, 0), (0, 1)],
                mode='constant', constant_values=1.)

    def __process_lidar_snapshots(self, frames):
        """Add semantic LIDAR points from measurements to collection of points
        with __sensor_transform_at_t0 as the origin.

        Parameters
        ==========
        frames : list of int
            The frames to merge LIDAR measurements to collection
            sorted by ascending order.
        """
        if not frames:
            return
        collection_of_points = []
        collection_of_labels = []
        collection_of_object_ids = []
        for frame in frames:
            lidar_measurement = self.__lidar_feeds[frame]
            raw_data = lidar_measurement.raw_data
            mtx = np.array(lidar_measurement.transform.get_matrix())
            data = np.frombuffer(raw_data, dtype=np.dtype([
                    ('x', np.float32), ('y', np.float32), ('z', np.float32),
                    ('CosAngle', np.float32), ('ObjIdx', np.uint32),
                    ('ObjTag', np.uint32)]))
            points = np.array([data['x'], data['y'], data['z']]).T
            labels = data['ObjTag']
            object_ids = data['ObjIdx']
            self.__lidar_snapshot_to_populate_vehicle_visibility(lidar_measurement,
                    points, labels, object_ids)
            points = self.__add_1_to_points(points)
            points = points @ mtx.T
            points = points[:, :3]
            collection_of_points.append(points)
            collection_of_labels.append(labels)
            collection_of_object_ids.append(object_ids)
        if self.__sensor_loc_at_t0 is None:
            lidar_measurement = self.__lidar_feeds[frames[0]]
            loc = carlautil.transform_to_location_ndarray(lidar_measurement.transform)
            self.__sensor_loc_at_t0 = loc
        else:
            collection_of_points     = [self.__overhead_points] + collection_of_points
            collection_of_labels     = [self.__overhead_labels] + collection_of_labels
            collection_of_object_ids = [self.__overhead_ids]    + collection_of_object_ids
        self.__overhead_points = np.concatenate(collection_of_points)
        self.__overhead_labels = np.concatenate(collection_of_labels)
        self.__overhead_ids    = np.concatenate(collection_of_object_ids)

    def __build_scene_data(self):
        """Does post/mid collection processing of the data.
        This method copies the data and does processing on the copy if 
        __scene_config is a SceneConfig, or directly processes the data if
        __scene_config is a OnlineConfig.

        Reflects all coordinates data about the x-axis in-place.
        This is needed because CARLA 0.9.11 uses a flipped y-axis.
        
        Returns
        =======
        np.array
            Current overhead points collected by scene builder of shape (# points, 3)
            that has been flipped about the x-axis.
        pd.DataFrame
            Current trajectory data collected by scene builder with relevant data
            (position, velocity, acceleration, heading) flipped about the x-axis.
        """
        scene_data = SceneBuilderData(self.__save_directory, self.__scene_name, 
                self.__map_reader.map_name,
                self.__world.get_settings().fixed_delta_seconds,
                self.__trajectory_data,
                self.__overhead_points, self.__overhead_labels,
                self.__overhead_ids,
                self.__map_reader.map_data,
                self.__vehicle_visibility, self.__sensor_loc_at_t0,
                self.__first_frame, self.__scene_config)
        
        if self.is_online:
            """Copies the scene data.
            WARNING: map_data, scene_config are passed as reference."""
            attr_names = ['trajectory_data', 'overhead_points', 'overhead_labels',
                    'overhead_ids', 'sensor_loc_at_t0', 'vehicle_visibility']
            for name in attr_names:
                setattr(scene_data, name, copy.deepcopy(getattr(scene_data, name)))

        """Reflect and return scene raw data."""
        scene_data.overhead_points[:, 1] *= -1
        scene_data.trajectory_data[['y', 'v_y', 'a_y']] *= -1
        scene_data.trajectory_data['heading'] = util.reflect_radians_about_x_axis(
                scene_data.trajectory_data['heading'])
        
        """Sort Trajectory data"""
        scene_data.trajectory_data.sort_values('frame_id', inplace=True)

        return scene_data

    def __checkpoint(self):
        """Do mid/post collection processing of the data and return
        the data as a SceneBuilderData.
        Also reflects all coordinates data along the x-axis."""

        """Finish LIDAR raw data"""
        if self.is_online:
            frames = set(self.__lidar_feeds.keys()) - self.__seen_lidar_keys
            self.__seen_lidar_keys.update(frames)
        else:
            frames = range(self.__first_frame, self.__last_frame + 1)
        frames = sorted(frames)
        self.__process_lidar_snapshots(frames)
        return self.__build_scene_data()

    def __remove_scene_builder(self):
        self.__data_collector.remove_scene_builder(self.__first_frame)

    @abstractmethod
    def process_scene(self, scene_data):
        """Process the scene.

        Parameters
        ==========
        scene_data : SceneBuilderData
            The data in process the scene with.
        
        Returns
        =======
        any
            Scene data to pass to the callback function.
        """
        pass

    def get_scene(self):
        """Get a scene composed of all the data collected so far.
        
        WARNING: This method should not be called
        if __scene_config is a SceneConfig.
        """

        logging.debug(f"in SceneBuilder.get_scene()")
        scene_data = self.__checkpoint()
        return self.process_scene(scene_data)

    def finish_scene(self):
        """Finish the scene. This function should be the *last* function
        to be called on the scene builder.

        SceneBuilder.finish_scene() allows the scene builder to call itself.
        This way the data collector does not have to figure out whether the
        scene builder has finished collecting all the data it needs.
        
        WARNING: This method should not be called
        if __scene_config is a OnlineConfig."""

        """Finish scene and removes scene builder from data collector. Calls process_scene()."""
        logging.debug(f"in SceneBuilder.finish_scene()")

        """Reach checkpoint in data collection before doing starting data processing."""
        scene_data = self.__checkpoint()

        """Process the scene and send the output to its destination via callback"""
        self.__callback(self.process_scene(scene_data))

        """Remove scene builder from data collector."""
        self.__remove_scene_builder()

    def capture_trajectory(self, frame):
        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
            should_capture_at_frame = True if self.is_online else \
                    frame_id < self.__scene_config.scene_interval
            if should_capture_at_frame:
                logging.debug(f"in SceneBuilder.capture_trajectory() frame_id = {frame_id}")
                self.__capture_agents_within_radius(frame_id)
            else:
                if self.__finished_lidar_trajectory:
                    self.finish_scene()

    def capture_lidar(self, lidar_measurement):
        frame = lidar_measurement.frame
        if frame >= self.__last_frame:
            self.__finished_lidar_trajectory = True


def points_to_2d_histogram(points, x_min, x_max, y_min, y_max, pixels_per_m):
    bins= [
            pixels_per_m*(x_max - x_min),
            pixels_per_m*(y_max - y_min)]
    range = [[x_min, x_max], [y_min, y_max]]
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=range)
    return hist


def round_to_int(x):
    return np.int(np.round(x))
