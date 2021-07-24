"""Functions and classes for generation of data."""

# Built-in libraries
import os
import collections
import abc
from abc import ABC, abstractmethod
import enum
import weakref
import logging

# PyPI libraries
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

# Local libararies
import carla
import utility as util
import carlautil
import carlautil.debug

from .map import MapQuerier, NaiveMapQuerier
from .map import Map10HDBoundTIntersectionReader, IntersectionReader
from .label import ScenarioIntersectionLabel, ScenarioSlopeLabel, BoundingRegionLabel
from .label import SampleLabelMap, SampleLabelFilter
from .label import SegmentationLabel, carla_id_maker
from .scene import SceneBuilder, SceneConfig
from .scene.v3.trajectron_scene import TrajectronPlusPlusSceneBuilder

def get_all_vehicle_blueprints(world):
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('t2')]
    return sorted(blueprints, key=lambda bp: bp.id)

def create_semantic_lidar_blueprint(world):
    """Construct a semantic LIDAR sensor blueprint.
    Based on https://ouster.com/products/os2-lidar-sensor/

    Note: semantic LIDAR does not have dropoff or noise.
    May have to mock this in the preprocessesing stage.

    """
    bp_library = world.get_blueprint_library()
    lidar_bp = bp_library.find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('channels', '128')
    lidar_bp.set_attribute('range', '70')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '10.0')
    lidar_bp.set_attribute('upper_fov', '0.0')
    lidar_bp.set_attribute('lower_fov', '-20.0')
    return lidar_bp


class AbstractDataCollector(ABC):
    """Data collector that holds scenes of instance SceneBuilder."""

    @abstractmethod
    def remove_scene_builder(self, frame):
        pass


class DataCollector(AbstractDataCollector):
    """Data collector based on DIM and Trajectron++."""

    Z_SENSOR_REL = 2.5

    def __create_segmentation_lidar_sensor(self):
        return self.__world.spawn_actor(
                create_semantic_lidar_blueprint(self.__world),
                carla.Transform(carla.Location(z=self.Z_SENSOR_REL)),
                attach_to=self.__ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid)

    def __init__(self, player_actor,
            map_reader,
            other_vehicle_ids,
            scene_config=SceneConfig(),
            scene_builder_cls=TrajectronPlusPlusSceneBuilder,
            save_frequency=35,
            save_directory='out',
            n_burn_frames=60,
            episode=0,
            exclude_samples=SampleLabelFilter(),
            callback=lambda x: x,
            debug=False):
        """
        Parameters
        ----------
        player_actor : carla.Vehicle
            Vehicle that the data collector is following around, and should collect data for.
        map_reader : MapQuerier
            One MapQuerier instance should be shared by all data collectors.
        record_interval : int
            Number of timesteps
        scene_interval : int

        save_frequency : int
            Frequency (wrt. to CARLA simulator frame) to save data.
        n_burn_frames : int
            The number of initial frames to skip before saving data.
        episode : int
            Index of current episode to collect data from.
        exclude_samples : SampleLabelFilter
            Filter to exclude saving samples by label.
        """
        self.__ego_vehicle = player_actor
        # __map_reader : MapQuerier
        self.__map_reader = map_reader
        self.__save_frequency = save_frequency
        self.__scene_config = scene_config
        self.__scene_builder_cls = scene_builder_cls
        self.__save_directory = save_directory
        self.n_burn_frames = n_burn_frames
        self.episode = episode
        self.__exclude_samples = exclude_samples
        self.__callback = callback
        self.__debug = debug

        # __make_scene_name : function
        #     Scene names (scene ID) are created using util.IDMaker
        self.__make_scene_name = lambda frame: carla_id_maker.make_id(
                map=self.__map_reader.map_name, episode=self.episode,
                agent=self.__ego_vehicle.id, frame=frame)

        self.__world = self.__ego_vehicle.get_world()

        # __scene_builders : map of (int, SceneBuilder)
        #     Store scene builders by frame.
        self.__scene_builders = {}

        vehicles = self.__world.get_actors(other_vehicle_ids)
        # __other_vehicles : list of carla.Vehicle
        #     List of IDs of vehicles not including __ego_vehicle.
        #     Use this to track other vehicles in the scene at each timestep. 
        self.__other_vehicles = dict(zip(other_vehicle_ids, vehicles))

        # __sensor : carla.Sensor
        #     Segmentation sensor. Data points will be used to construct overhead.
        self.__sensor = self.__create_segmentation_lidar_sensor()

        # __n_feeds : int
        #     Size of LIDAR feed dict
        self.__n_feeds = (self.__scene_config.scene_interval \
                + 1)*self.__scene_config.record_interval

        # __first_frame : int
        #     First frame in simulation. Used to find current timestep.
        self.__first_frame = None

        # __lidar_feeds : collections.OrderedDict
        #     Where int key is frame index and value
        #     is a carla.LidarMeasurement or carla.SemanticLidarMeasurement
        self.__lidar_feeds = collections.OrderedDict()
        
    @property
    def n_scene_builders(self):
        return len(self.__scene_builders)

    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.__sensor.listen(lambda image: DataCollector.parse_image(weak_self, image))
    
    def stop_sensor(self):
        """Stop the sensor."""
        self.__sensor.stop()

    @property
    def sensor_is_listening(self):
        return self.__sensor.is_listening

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.__sensor.destroy()
        self.__sensor = None

    def remove_scene_builder(self, frame):
        if frame in self.__scene_builders:
            del self.__scene_builders[frame]

    def finish_scenes(self):
        for scene_builder in list(self.__scene_builders.values()):
            scene_builder.finish_scene()
    
    def __should_create_scene_builder(self, frame):
        return (frame - self.__first_frame - self.n_burn_frames) \
                % (self.__scene_config.record_interval * self.__save_frequency) == 0

    def capture_step(self, frame):
        if self.__first_frame is None:
            self.__first_frame = frame
        if frame - self.__first_frame < self.n_burn_frames:
            return
        if self.__should_create_scene_builder(frame):
            logging.debug(f"in DataCollector.capture_step() player = {self.__ego_vehicle.id} frame = {frame}")
            logging.debug("Create scene builder")
            self.__scene_builders[frame] = self.__scene_builder_cls(self,
                    self.__map_reader,
                    self.__ego_vehicle,
                    self.__other_vehicles,
                    self.__lidar_feeds,
                    self.__make_scene_name(frame),
                    frame,
                    scene_config=self.__scene_config,
                    save_directory=self.__save_directory,
                    exclude_samples=self.__exclude_samples,
                    callback=self.__callback,
                    debug=self.__debug)
        for scene_builder in list(self.__scene_builders.values()):
            scene_builder.capture_trajectory(frame)
        
        while len(self.__lidar_feeds) > self.__n_feeds:
            self.__lidar_feeds.popitem(last=False)

    @staticmethod
    def parse_image(weak_self, image):
        """Pass sensor image to each scene builder.

        Parameters
        ==========
        image : carla.SemanticLidarMeasurement
        """
        self = weak_self()
        if not self:
            return
        logging.debug(f"in DataCollector.parse_image() player = {self.__ego_vehicle.id} frame = {image.frame}")
        self.__lidar_feeds[image.frame] = image
        for scene_builder in list(self.__scene_builders.values()):
            scene_builder.capture_lidar(image)

