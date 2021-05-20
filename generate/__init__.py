"""Functions and classes for data collection."""

import os
import collections
import abc
from abc import ABC, abstractmethod
import enum
import weakref
import logging

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import carla

import utility as util
import carlautil
import carlautil.debug

class SceneConfig(object):
    """Configuration used by scene builder."""
    def __init__(self, scene_interval=32,
            record_interval=5,
            pixel_dim=0.5):
        self.scene_interval = scene_interval
        self.record_interval = record_interval
        self.pixel_dim = pixel_dim

class ScenarioIntersectionLabel(object):
    """Labels samples by proximity of vehicle to intersections."""

    # NONE : str
    #     Vehicle is not near any intersections
    NONE = 'NONE'
    # UNCONTROLLED : str
    #     Vehicle is near an uncontrolled intersection
    UNCONTROLLED = 'UNCONTROLLED'
    # CONTROLLED : str
    #     Vehicle is near a controlled intersection
    CONTROLLED = 'CONTROLLED'

class ScenarioSlopeLabel(object):
    """Labels samples by proximity of vehicle to slopes."""

    # NONE : str
    #     Vehicle is not near any intersections
    NONE = 'NONE'
    # SLOPES : str
    #     Vehicle is close or on a sloped road
    SLOPES = 'SLOPES'

class BoundingRegionLabel(object):
    """Labels samples whether they are inside a bounding region.
    Use this to select cars on (a) specific lane(s), intersection(s)"""

    # NONE : str
    #     Vehicle is not inside any bounding region
    NONE = 'NONE'
    # BOUNDED : str
    #     Vehicle is inside a bounding region
    BOUNDED = 'BOUNDED'

class SampleLabelMap(object):
    """Container of sample labels, categorized by different types."""
    
    def __init__(self,
            intersection_type=ScenarioIntersectionLabel.NONE,
            slope_type=ScenarioSlopeLabel.NONE,
            bounding_type=BoundingRegionLabel.NONE,
            slope_pitch=0.0):
        self.intersection_type = intersection_type
        self.slope_type = slope_type
        self.bounding_type = bounding_type
        self.slope_pitch = slope_pitch

class SampleLabelFilter(object):
    """Container for sample label filter."""

    def __init__(self,
            intersection_type=[],
            slope_type=[],
            bounding_type=[]):
        """
        Parameters
        ----------
        intersection_type : list of str
        slope_type : list of str
        """
        self.intersection_type = intersection_type
        self.slope_type = slope_type
        self.bounding_type = bounding_type

    def contains(self, _type, label):
        """Check whether a label of type _type is in the filter.

        Parameters
        ----------
        _type : str
            Label type to lookup.
        label : str
            Label to check for existence in filter.

        Returns
        -------
        bool
        """
        return label in getattr(self, _type, [])

def create_semantic_lidar_blueprint(world):
    """Construct a semantic LIDAR sensor blueprint.

    Note: semantic LIDAR does not have dropoff or noise.
    May have to mock this in the preprocessesing stage.
    """
    bp_library = world.get_blueprint_library()
    lidar_bp = bp_library.find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '10.0')
    lidar_bp.set_attribute('upper_fov', '10.0')
    lidar_bp.set_attribute('lower_fov', '-30.0')
    return lidar_bp

class MapQuerier(ABC):
    """Abstract class to keep track of properties in a map and
    used to query whether an actor is in a certain part
    (i.e. intersection, hill) of the map for the sake of
    labeling samples."""

    def __init__(self, carla_world, carla_map, debug=False):
        self.carla_world = carla_world
        self.carla_map = carla_map
        self._debug = debug

    @property
    def map_name(self):
        return self.carla_map.name

    def debug_display(self):
        pass

    def at_intersection_to_label(self, actor):
        """Retrieve the label corresponding to the actor's location in the
        map based on proximity to intersections."""
        return ScenarioIntersectionLabel.NONE

    def at_slope_to_label(self, actor):
        """Check wheter actor (i.e. vehicle) is near a slope,
        returning a ScenarioSlopeLabel."""
        return ScenarioSlopeLabel.NONE, 0

    def at_bounding_box_to_label(self, actor):
        """Check whether actor (i.e. vehicle) is inside a bounding region."""
        return BoundingRegionLabel.NONE
    
    def get_actor_labels(self, actor):
        slope_type, slope_pitch = self.at_slope_to_label(actor)
        return SampleLabelMap(
                intersection_type=self.at_intersection_to_label(actor),
                slope_type=slope_type,
                bounding_type=self.at_bounding_box_to_label(actor),
                slope_pitch=slope_pitch)

class NaiveMapQuerier(MapQuerier):
    pass

class Map10HDBoundTIntersectionReader(MapQuerier):
    """Label samples in Map10HD that are in specific
    T-intersection."""

    REGION_RADIUS = 18
    CENTER_COORDS = np.array([42.595, 65.077, 0.0])

    def __init__(self, carla_world, carla_map, debug=False):
        """
        """
        super().__init__(carla_world, carla_map, debug=debug)

    def debug_display(self):
        display_time = 40.0
        x, y, z = self.CENTER_COORDS
        carlautil.debug.show_square(self.carla_world, x, y, z,
                self.REGION_RADIUS, t=display_time)

    def at_bounding_box_to_label(self, actor):
        actor_location = carlautil.actor_to_location_ndarray(actor)
        dist = np.linalg.norm(self.CENTER_COORDS - actor_location)
        if dist < self.REGION_RADIUS:
            return BoundingRegionLabel.BOUNDED
        else:
            return BoundingRegionLabel.NONE


class IntersectionReader(MapQuerier):
    """Used to query whether actor is on an intersection and a hill on the map.
    """

    # degree threshold for sloped road.
    SLOPE_DEGREES = 5.0
    SLOPE_UDEGREES = (360. - SLOPE_DEGREES)
    SLOPE_FIND_RADIUS = 30
    # radius used to check whether junction has traffic lights
    TLIGHT_FIND_RADIUS = 25
    # radius used to check whether vehicle is in junction
    TLIGHT_DETECT_RADIUS = 28

    def __init__(self, carla_world, carla_map, debug=False):
        """
        Parameters
        ----------
        carla_world : carla.World
        carla_map : carla.Map
        debug : bool
        """
        super().__init__(carla_world, carla_map, debug=debug)
        """Generate slope topology of the map"""
        # get waypoints
        wps = self.carla_map.generate_waypoints(4.0)
        def h(wp):
            l = wp.transform.location
            pitch = wp.transform.rotation.pitch
            return np.array([l.x, l.y, l.z, pitch])
        loc_and_pitch_of_wps = util.map_to_ndarray(h, wps)
        self.wp_locations = loc_and_pitch_of_wps[:, :-1]
        self.wp_pitches = loc_and_pitch_of_wps[:, -1]
        self.wp_pitches = self.wp_pitches % 360.
        self.wp_is_sloped = np.logical_and(
                self.SLOPE_DEGREES < self.wp_pitches,
                self.wp_pitches < self.SLOPE_UDEGREES)
        
        """Generate intersection topology of the map"""
        # get nodes in graph
        topology = self.carla_map.get_topology()
        G = nx.Graph()
        G.add_edges_from(topology)
        tlights = util.filter_to_list(lambda a: 'traffic_light' in a.type_id,
                self.carla_world.get_actors())
        junctions = carlautil.get_junctions_from_topology_graph(G)

        tlight_distances = np.zeros((len(tlights), len(junctions),))
        f = lambda j: carlautil.location_to_ndarray(j.bounding_box.location)
        junction_locations = util.map_to_ndarray(f, junctions)
        
        g = lambda tl: carlautil.transform_to_location_ndarray(
                tl.get_transform())
        tlight_locations = util.map_to_ndarray(g, tlights)

        for idx, junction in enumerate(junctions):
            tlight_distances[:,idx] = np.linalg.norm(
                    tlight_locations - junction_locations[idx], axis=1)

        is_controlled_junction = (tlight_distances < self.TLIGHT_FIND_RADIUS).any(axis=0)
        is_uncontrolled_junction = np.logical_not(is_controlled_junction)
        self.controlled_junction_locations \
                = junction_locations[is_controlled_junction]
        self.uncontrolled_junction_locations \
                = junction_locations[is_uncontrolled_junction]

    def debug_display(self):
        display_time = 40.0
        for loc in self.controlled_junction_locations:
            self.carla_world.debug.draw_string(
                    carlautil.ndarray_to_location(loc) + carla.Location(z=3.0),
                    'o',
                    color=carla.Color(r=255, g=0, b=0, a=100),
                    life_time=display_time)
        for loc in self.uncontrolled_junction_locations:
            self.carla_world.debug.draw_string(
                    carlautil.ndarray_to_location(loc) + carla.Location(z=3.0),
                    'o',
                    color=carla.Color(r=0, g=255, b=0, a=100),
                    life_time=display_time)

    def at_slope_to_label(self, actor):
        """Check wheter actor (i.e. vehicle) is near a slope,
        returning a ScenarioSlopeLabel.

        TODO: make wp_locations array size smaller after debugging.
        Don't need to check non-sloped waypoints.
        """
        actor_location = carlautil.actor_to_location_ndarray(actor)
        actor_xy = actor_location[:2]
        actor_z = actor_location[-1]
        """Want to ignore waypoints above and below (i.e. in the case actor is on a bridge)."""
        upperbound_z = actor.bounding_box.extent.z * 2
        lowerbound_z = -1
        xy_distances_to_wps = np.linalg.norm(
            self.wp_locations[:, :2] - actor_xy, axis=1)
        z_displacement_to_wps = self.wp_locations[:, -1] - actor_z

        """get waypoints close to vehicle filter"""
        wps_filter = np.logical_and(
                xy_distances_to_wps < self.SLOPE_FIND_RADIUS,
                np.logical_and(
                    z_displacement_to_wps < upperbound_z,
                    z_displacement_to_wps > lowerbound_z))

        #####
        """obtain extra slope information"""
        wp_pitches = self.wp_pitches[wps_filter]
        if wp_pitches.size == 0:
            max_wp_pitch = 0.0
        else:
            wp_pitches = np.min(
                    np.vstack((wp_pitches, np.abs(wp_pitches - 360.),)),
                    axis=0)
            max_wp_pitch = np.max(wp_pitches)
        #####

        if self._debug:
            nearby_slopes = np.logical_and(
                    self.wp_is_sloped == True,
                    wps_filter)
            for wp_location in self.wp_locations[nearby_slopes]:
                loc = carlautil.ndarray_to_location(wp_location)
                carlautil.debug_point(self.carla_world, loc)

        if np.any(self.wp_is_sloped[wps_filter]):
            return ScenarioSlopeLabel.SLOPES, max_wp_pitch
        else:
            return ScenarioSlopeLabel.NONE, max_wp_pitch

    def at_intersection_to_label(self, actor):
        """Retrieve the label corresponding to the actor's location in the
        map based on proximity to intersections.
        
        Parameters
        ----------
        actor : carla.Actor

        Returns
        """
        actor_location = carlautil.actor_to_location_ndarray(actor)
        distances_to_uncontrolled = np.linalg.norm(
                self.uncontrolled_junction_locations - actor_location, axis=1)
        if np.any(distances_to_uncontrolled < self.TLIGHT_DETECT_RADIUS):
            return ScenarioIntersectionLabel.UNCONTROLLED
        distances_to_controlled = np.linalg.norm(
                self.controlled_junction_locations - actor_location, axis=1)
        if np.any(distances_to_controlled < self.TLIGHT_DETECT_RADIUS):
            return ScenarioIntersectionLabel.CONTROLLED
        return ScenarioIntersectionLabel.NONE


class SegmentationLabel(enum.Enum):
    RoadLine = 6
    Road = 7
    SideWalk = 8
    Vehicles = 10


class SceneBuilder(object):
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
            scene_radius=200,
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
        self.scene_name = scene_name
        # __first_frame : int
        self.__first_frame = first_frame
        self.__world = self.__ego_vehicle.get_world()
        self.__save_directory = save_directory
        
        self.__exclude_samples = exclude_samples
        self.__debug = debug

        self.__scene_config = scene_config
        self.__scene_count = 0
        self.__last_frame = self.__first_frame + (self.__scene_config.scene_interval \
                - 1)*self.__scene_config.record_interval + 1

        self.__finished_capture_trajectory = False
        self.__finished_lidar_trajectory = False

        # __other_vehicle_visibility : map of (int, set)
        self.__other_vehicle_visibility = { }

        self.__radius = scene_radius
        # __sensor_loc_at_t0 : carla.Transform
        self.__sensor_loc_at_t0 = None
        # __overhead_points : np.float32
        self.__overhead_points = None
        # __overhead_point_labels : np.uint32
        self.__overhead_point_labels = None
        # __overhead_point_ids : np.uint32
        self.__overhead_point_ids = None
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
        """
        TODO: combine with trajectron-plus-plus code
        """
        labels = self.__map_reader.get_actor_labels(self.__ego_vehicle)
        if self.__should_exclude_dataset_sample(labels):
            self.__remove_scene()
        player_location = carlautil.actor_to_location_ndarray(self.__ego_vehicle)
        if len(self.__other_vehicles):
            other_ids = self.__other_vehicles.keys()
            other_vehicles = self.__other_vehicles.values()
            others_data = carlautil.vehicles_to_xyz_lwh_pyr_ndarray(other_vehicles).T
            other_locations = others_data[:,:3]
            distances = np.linalg.norm(other_locations - player_location, axis=1)
            df = pd.DataFrame({
                    'frame_id': np.full((len(other_ids),), frame_id),
                    'type': ['VEHICLE'] * len(other_ids), # use env.NodeType.VEHICLE
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
            df = df[df['distances'] < self.radius]
            df = df[df['z'].between(self.Z_LOWERBOUND, self.Z_UPPERBOUND, inclusive=False)]
            del df['distances']
        else:
            df = pd.DataFrame(columns=['frame_id', 'type', 'node_id', 'robot',
                    'x', 'y', 'z', 'length', 'width', 'height', 'heading'])
        ego_data = carlautil.vehicle_to_xyz_lwh_pyr_ndarray(self.__ego_vehicle)
        data_point = pd.Series({
                'frame_id': frame_id,
                'type': 'VEHICLE', # use env.NodeType.VEHICLE
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
            vehicle_label_mask = labels == SegmentationLabel.Vehicles.value
            object_ids = object_ids[vehicle_label_mask]
            self.__other_vehicle_visibility[frame_id] = set(object_ids)

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
        
    def __process_lidar(self):
        for frame in range(self.__first_frame, self.__last_frame + 1):
            lidar_measurement = self.__lidar_feeds[frame]
            self.__process_lidar_snapshot(lidar_measurement)

    def remove_scene(self):
        self.__data_collector.remove_scene_builder(self.__first_frame)

    def finish_scene(self):
        """
        TODO: Refactor SceneBuilder so I can subclass it based on finish_scene() implementation.
        """

        self.remove_scene()
        # self.__trajectory_data.sort_values('frame_id', inplace=True)
        # print(self.__trajectory_data)

        # should need to adjust points based on rel. sensor loc. from ego vehicle loc.
        # self.__data_collector.Z_SENSOR_REL

        # Process all of the LIDAR points
        self.__process_lidar()
        # Select road LIDAR points.
        road_label_mask = self.__overhead_labels == SegmentationLabel.Road.value
        points = self.__overhead_points[road_label_mask]
        # Trim points above/below certain Z levels.
        z_mask = np.logical_and(
                points[:, 2] > self.Z_LOWERBOUND, points[:, 2] < self.Z_UPPERBOUND)
        points = points[z_mask]

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot()
        ax.scatter(points[:, 0], points[:, 1], s=2, c='blue')
        ax.scatter(self.__trajectory_data['x'], self.__trajectory_data['y'], s=2, c='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fn = f"out.png"
        fp = os.path.join(self.__save_directory, fn)
        fig.savefig(fp)

    def capture_trajectory(self, frame):
        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
            if frame_id < self.__scene_config.scene_interval:
                logging.info(f"in SceneBuilder.capture_trajectory() frame_id = {frame_id}")
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


class DataCollector(object):
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
            save_frequency=35,
            save_directory='out',
            n_burn_frames=60,
            episode=0,
            exclude_samples=SampleLabelFilter(),
            debug=False):
        """
        Parameters
        ----------
        player_actor : carla.Vehicle
            Vehicle that the data collector is following around, and should collect data for.
        map_reader : IntersectionReader
            One IntersectionReader instance should be shared by all data collectors.
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
        self.__map_reader = map_reader
        self.__save_frequency = save_frequency
        self.__scene_config = scene_config
        self.__save_directory = save_directory
        self.n_burn_frames = n_burn_frames
        self.episode = episode
        self.__exclude_samples = exclude_samples
        self.__debug = debug

        # __make_scene_name : function
        #     Scene names are episode/agent/frame
        self.__make_scene_name = lambda frame : "{}/ep{:03d}/agent{:03d}/frame{:08d}".format(
                self.__map_reader.map_name, self.episode, self.__ego_vehicle.id, frame)

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
            logging.info(f"in DataCollector.capture_step() player = {self.__ego_vehicle.id} frame = {frame}")
            logging.info("Create scene builder")
            self.__scene_builders[frame] = SceneBuilder(self,
                    self.__map_reader,
                    self.__ego_vehicle,
                    self.__other_vehicles,
                    self.__lidar_feeds,
                    self.__make_scene_name(frame),
                    frame,
                    scene_config=self.__scene_config,
                    save_directory=self.__save_directory,
                    exclude_samples=self.__exclude_samples,
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
        logging.info(f"in DataCollector.parse_image() player = {self.__ego_vehicle.id} frame = {image.frame}")
        self.__lidar_feeds[image.frame] = image
        for scene_builder in list(self.__scene_builders.values()):
            scene_builder.capture_lidar(image)
