import os
import collections
from abc import ABC, abstractmethod
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

class SceneConfig(object):
    """Configuration used by scene builder."""
    def __init__(self, simulation_dt=0.1,
            scene_interval=32,
            record_interval=5,
            pixels_per_m=3,
            node_type=NodeTypeEnum(['VEHICLE'])):
        self.simulation_dt = simulation_dt # not being used?
        self.scene_interval = scene_interval
        self.record_interval = record_interval
        self.pixels_per_m = pixels_per_m
        self.node_type = node_type


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
    """Constructs a scene that contains scene_interval samples
    of consecutive snapshots of the simulation. The scenes have:

    Vehicle bounding boxes and locations in cartesion coordinates.

    LIDAR overhead bitmap consisting of LIDAR points collected at
    each of the scene_interval timesteps by the ego vehicle.
    """

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

        self.__scene_config = scene_config
        self.__last_frame = self.__first_frame + (self.__scene_config.scene_interval \
                - 1)*self.__scene_config.record_interval + 1

        self.__finished_capture_trajectory = False
        self.__finished_lidar_trajectory = False

        # __vehicle_visibility : map of (int, set of int)
        #    IDs of vehicles visible to the ego vehicle by frame ID.
        #    Vehicles are visible when then are hit by semantic LIDAR.
        self.__vehicle_visibility = { }

        self.__radius = scene_radius
        # __sensor_loc_at_t0 : carla.Transform
        self.__sensor_loc_at_t0 = None
        # __overhead_points : np.float32
        self.__overhead_points = None
        # __overhead_point_labels : np.uint32
        self.__overhead_point_labels = None
        # __overhead_ids : np.uint32
        self.__overhead_ids = None
        # __trajectory_data : pd.DataFrame
        self.__trajectory_data = None
    
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
            others_data = carlautil.vehicles_to_xyz_lwh_pyr_ndarray(other_vehicles).T
            other_locations = others_data[:3].T
            distances = np.linalg.norm(other_locations - player_location, axis=1)
            df = pd.DataFrame({
                    'frame_id': np.full((len(other_ids),), frame_id),
                    # use env.NodeType.VEHICLE
                    'type': [self.__scene_config.node_type.VEHICLE] * len(other_ids),
                    'node_id': other_ids,
                    'robot': [False] * len(other_ids),
                    'distances': distances,
                    'x': others_data[0],
                    'y': others_data[1],
                    'z': others_data[2],
                    'length': others_data[3],
                    'width': others_data[4],
                    'height': others_data[5],
                    'heading': others_data[7]})
            df = df[df['distances'] < self.__radius]
            df = df[df['z'].between(self.Z_LOWERBOUND, self.Z_UPPERBOUND, inclusive=False)]
            del df['distances']
        else:
            df = pd.DataFrame(columns=['frame_id', 'type', 'node_id', 'robot',
                    'x', 'y', 'z', 'length', 'width', 'height', 'heading'])
        ego_data = carlautil.vehicle_to_xyz_lwh_pyr_ndarray(self.__ego_vehicle)
        data_point = pd.Series({
                'frame_id': frame_id,
                # use env.NodeType.VEHICLE
                'type': self.__scene_config.node_type.VEHICLE,
                'node_id': 'ego',
                'robot': True,
                'x': ego_data[0],
                'y': ego_data[1],
                'z': ego_data[2],
                'length': ego_data[3],
                'width': ego_data[4],
                'height': ego_data[5],
                'heading': ego_data[7]})
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
            object_ids = object_ids[vehicle_label_mask]
            self.__vehicle_visibility[frame_id] = set(object_ids)

    def __add_1_to_points(self, points):
        return np.pad(points, [(0, 0), (0, 1)],
                mode='constant', constant_values=1.)

    def __process_lidar_snapshot(self, lidar_measurement):
        """Add semantic LIDAR points from measurement to a collection of points
        with __sensor_transform_at_t0 as the origin.

        Parameters
        ==========
        lidar_measurement : carla.SemanticLidarMeasurement
        """
        raw_data = lidar_measurement.raw_data
        mtx = np.array(lidar_measurement.transform.get_matrix())
        data = np.frombuffer(raw_data, dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
        points = np.array([data['x'], data['y'], data['z']]).T
        labels = data['ObjTag']
        object_ids = data['ObjIdx']
        self.__lidar_snapshot_to_populate_vehicle_visibility(lidar_measurement,
                points, labels, object_ids)
        points = self.__add_1_to_points(points)
        points = (mtx @ points.T).T
        points = points[:, :3]
        if self.__sensor_loc_at_t0 is None:
            loc = carlautil.transform_to_location_ndarray(lidar_measurement.transform)
            self.__sensor_loc_at_t0 = loc
            self.__overhead_points = points
            self.__overhead_labels = labels
            self.__overhead_ids = object_ids
        else:
            self.__overhead_points = np.concatenate((self.__overhead_points, points))
            self.__overhead_labels = np.concatenate((self.__overhead_labels, labels))
            self.__overhead_ids = np.concatenate((self.__overhead_ids, object_ids))
        
    def __complete_data_collection(self):
        """
        """
        # Remove scene builder from data collector.
        self.__remove_scene_builder()
        # Finish LIDAR raw data
        for frame in range(self.__first_frame, self.__last_frame + 1):
            lidar_measurement = self.__lidar_feeds[frame]
            self.__process_lidar_snapshot(lidar_measurement)
        # Finish Trajectory raw data
        for l in self.__vehicle_visibility.values():
            l.add('ego')
        self.__trajectory_data.sort_values('frame_id', inplace=True)
        # Return scene raw data
        return SceneBuilderData(self.__save_directory, self.__scene_name, 
                self.__map_reader.map_name,
                self.__world.get_settings().fixed_delta_seconds,
                self.__trajectory_data,
                self.__overhead_points, self.__overhead_labels,
                self.__overhead_ids,
                self.__map_reader.map_data,
                self.__vehicle_visibility, self.__sensor_loc_at_t0,
                self.__first_frame, self.__scene_config)

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

    def finish_scene(self):
        """Finish scene and removes scene builder from data collector. Calls process_scene()."""
        logging.debug(f"in SceneBuilder.finish_scene()")
        # Complete data collection before doing starting data processing.
        data = self.__complete_data_collection()
        self.__callback(self.process_scene(data))

    def capture_trajectory(self, frame):
        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
            if frame_id < self.__scene_config.scene_interval:
                logging.debug(f"in SceneBuilder.capture_trajectory() frame_id = {frame_id}")
                self.__capture_agents_within_radius(frame_id)
            else:
                if self.__finished_lidar_trajectory:
                    self.finish_scene()
                else:
                    self.__finished_capture_trajectory = True

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
