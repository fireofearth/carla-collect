"""Functions and classes for data collection.
"""

import collections
import abc
from abc import ABC, abstractmethod
import weakref
import logging
import attrdict
import networkx as nx
import numpy as np
import tensorflow as tf
import carla

import generate.overhead as generate_overhead
import generate.observation as generate_observation
import utility as util
import carlautil
import carlautil.debug
import precog.utils.class_util as classu
import precog.utils.tensor_util as tensoru

DEFAULT_PHI_ATTRIBUTES = attrdict.AttrDict({
            "T": 20, "T_past": 10, "B": 1, "A": 5,
            "C": 4, "D": 2, "H": 200, "W": 200})

class LidarParams(object):
    @classu.member_initialize
    def __init__(self,
            meters_max=50,
            pixels_per_meter=2,
            hist_max_per_pixel=25,
            val_obstacle=1.):
        pass

class ESPPhiData(object):
    @classu.member_initialize
    def __init__(self,
            S_past_world_frame=None,
            S_future_world_frame=None,
            yaws=None,
            overhead_features=None,
            agent_presence=None,
            light_strings=None):
        pass

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
    
    @classu.member_initialize
    def __init__(self,
            intersection_type=ScenarioIntersectionLabel.NONE,
            slope_type=ScenarioSlopeLabel.NONE,
            bounding_type=BoundingRegionLabel.NONE,
            slope_pitch=0.0,
            n_present="NONE"):
        pass

class SampleLabelFilter(object):
    """Container for sample label filter."""

    @classu.member_initialize
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
        pass

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

def create_phi(settings):
    s = settings
    tf.compat.v1.reset_default_graph()
    S_past_world_frame = tf.zeros(
            (s.B, s.A, s.T_past, s.D),
            dtype=tf.float64, name="S_past_world_frame") 
    S_future_world_frame = tf.zeros(
            (s.B, s.A, s.T, s.D),
            dtype=tf.float64, name="S_future_world_frame")
    yaws = tf.zeros(
            (s.B, s.A),
            dtype=tf.float64, name="yaws")
    overhead_features = tf.zeros(
            (s.B, s.H, s.W, s.C),
            dtype=tf.float64, name="overhead_features")
    agent_presence = tf.zeros(
            (s.B, s.A),
            dtype=tf.float64, name="agent_presence")
    light_strings = tf.zeros(
            (s.B,),
            dtype=tf.string, name="light_strings")
    return ESPPhiData(
            S_past_world_frame=S_past_world_frame,
            S_future_world_frame=S_future_world_frame,
            yaws=yaws,
            overhead_features=overhead_features,
            agent_presence=agent_presence,
            light_strings=light_strings)

def create_lidar_blueprint(world):
    bp_library = world.get_blueprint_library()
    """
    sensor.lidar.ray_cast creates a carla.LidarMeasurement per step

    attributes for sensor.lidar.ray_cast
    https://carla.readthedocs.io/en/latest/ref_sensors/#lidar-sensor

    doc for carla.SensorData
    https://carla.readthedocs.io/en/latest/python_api/#carla.SensorData

    doc for carla.LidarMeasurement
    https://carla.readthedocs.io/en/latest/python_api/#carla.LidarMeasurement
    """
    lidar_bp = bp_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '10.0')
    lidar_bp.set_attribute('upper_fov', '10.0')
    lidar_bp.set_attribute('lower_fov', '-30.0')
    return lidar_bp

def create_lidar_blueprint_v2(world):
    """Construct a stronger LIDAR sensor blueprint.
    Used for debugging.
    """
    bp_library = world.get_blueprint_library()
    lidar_bp = bp_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '48')
    lidar_bp.set_attribute('range', '60')
    lidar_bp.set_attribute('dropoff_general_rate', '0.38')
    lidar_bp.set_attribute('atmosphere_attenuation_rate', '0.003')
    lidar_bp.set_attribute('points_per_second', '100000')
    lidar_bp.set_attribute('rotation_frequency', '10.0')
    lidar_bp.set_attribute('upper_fov', '10.0')
    lidar_bp.set_attribute('lower_fov', '-30.0')
    return lidar_bp

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

    Note: bad naming. Class should be named MapReader or something more general.
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


class SceneBuilder(object):
    """Constructs a scene that contains scene_interval samples
    of consecutive snapshots of the simulation. The scenes have:

    Vehicle bounding boxes and locations in cartesion coordinates.

    LIDAR overhead bitmap consisting of LIDAR points collected at
    each of the scene_interval timesteps by the ego vehicle.
    """
    
    def __init__(self, data_collector,
            ego_vehicle,
            other_vehicles,
            make_sample_name,
            first_frame,
            scene_interval=32,
            record_interval=5,
            save_directory='out',
            pixel_dim=0.5,
            debug=False):
        """
        pixel_dim : np.array
            Dimension of a pixel in meters (m)
        """
        self.__data_collector = data_collector
        # __ego_vehicle : carla.Vehicle
        self.__ego_vehicle = ego_vehicle
        self.__save_directory = save_directory
        self.__make_sample_name = make_sample_name
        self.__debug = debug

        self.__first_frame = first_frame
        self.__scene_interval = scene_interval
        self.__record_interval = record_interval
        self.__scene_count = 0
        self.__lidar_frames_to_get = set(range(self.__first_frame,
                self.__first_frame \
                + (self.__scene_interval - 1)*self.__record_interval + 1))
        self.__lidar_frames_obtained = set()
        self.__finished_capture_trajectory = False
        self.__finished_lidar_trajectory = False

        self.__radius = 200
        # __other_vehicles : list of carla.Vehicle
        self.__other_vehicles = other_vehicles
        # __pixel_dim : float
        self.__pixel_dim = pixel_dim
        # __sensor_transform_at_t0 : carla.Transform
        self.__sensor_transform_at_t0 = None
        # __overhead_points : np.float32
        self.__overhead_points = None
        # __overhead_point_labels : np.uint32
        self.__overhead_point_labels = None
        # __overhead_point_ids : np.uint32
        self.__overhead_point_ids = None
        # __trajectory_data : pd.DataFrame
        self.__trajectory_data = None
    
    def __capture_agents_within_radius(self, frame):
        """
        TODO: add type, robot to data frame
        """
        player_location = carlautil.actor_to_location_ndarray(self.__ego_vehicle)
        other_ids = self.__other_vehicles.keys()
        other_vehicles = self.__other_vehicles.values()
        others_data = carlautil.vehicles_to_xyz_lwh_pyr_ndarray(other_vehicles)
        other_locations = others_data[:3]
        distances = np.linalg.norm(other_locations - player_location, axis=1)
        df = pd.DataFrame({
                'frame_id': np.full((len(other_ids),), self.__scene_count),
                'node_id': other_ids,
                'distances': distances,
                'x': others_data[0],
                'y': others_data[1],
                'z': others_data[2],
                'length': others_data[3],
                'width': others_data[4],
                'height': others_data[5],
                'heading': others_data[7]})
        upperbound_z = 4
        lowerbound_z = -4
        df = df[df['distances'] < self.radius]
        df = df[df['z'].between(lowerbound_z, upperbound_z, inclusive=False)]
        if self.__trajectory_data is None:
            self.__trajectory_data = df
        else:
            self.__trajectory_data = pd.concatenate((self.__trajectory_data, df))

    def __finish_scene(self):
        raise NotImplementedError()
        self.__data_collector.remove_scene_builder(self.__first_frame)

    def capture_trajectory(self, frame):
        if self.__scene_count < self.__scene_interval:
            self.__capture_agents_within_radius(frame)
        elif self.__scene_count >= self.__scene_interval:
            if self.__finished_capture_trajectory and self.__finished_lidar_trajectory:
                self.__finish_scene()
            else:
                self.__finished_capture_trajectory = True
        else:
            raise Exception("Scene builder has already finished building the scene.")
        self.__scene_count += 1
    
    def __capture_lidar_snapshot(self, lidar_measurement):
        """Add LIDAR points from measurement to a collection of points
        with __sensor_transform_at_t0 as the origin.

        Parameters
        ==========
        lidar_measurement : carla.SemanticLidarMeasurement
        """
        raw_data = lidar_measurement.raw_data
        data = np.frombuffer(raw_data, dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
        points = np.array([data['x'], data['y'], data['z']]).T
        labels = data['ObjTag']
        object_ids = data['ObjIdx']
        if
        if self.__sensor_transform_at_t0 is None:
            loc = carlautil.transform_to_location_ndarray(lidar_measurement.transform)
            mtx = carlautil.create_translation_mtx(loc[0], loc[1], loc[2])
            points = (mtx @ points.T).T
            self.__sensor_loc_at_t0 = loc
            self.__overhead_points = points
            self.__overhead_labels = labels
            self.__overhead_ids = object_ids
        else:
            loc = carlautil.transform_to_location_ndarray(lidar_measurement.transform)
            loc = loc - self.__sensor_loc_at_t0
            mtx = carlautil.create_translation_mtx(loc[0], loc[1], loc[2])
            points = (mtx @ points.T).T
            self.__overhead_points = np.concatenate((self.__overhead_points, points))
            self.__overhead_labels = np.concatenate((self.__overhead_labels, labels))
            self.__overhead_ids = np.concatenate((self.__overhead_ids, object_ids))

    def capture_lidar(self, lidar_measurement):
        frame = lidar_measurement.frame
        if frame in self.__lidar_frames_to_get \
                and frame not in self.__lidar_frames_obtained:
            self.__capture_lidar_snapshot(lidar_measurement)
            self.__lidar_frames_obtained.add(frame)
        if not self.__lidar_frames_to_get - self.__lidar_frames_obtained:
            if self.__finished_capture_trajectory and self.__finished_lidar_trajectory:
                self.__finish_scene()
            else:
                self.__finished_lidar_trajectory = True


class DataCollector(object):
    """Data collector based on DIM and Trajectron++."""

    def __create_segmentation_lidar_sensor(self):
        return self._world.spawn_actor(
                create_semantic_lidar_blueprint(self._world),
                carla.Transform(carla.Location(z=2.5)),
                attach_to=self.__player,
                attachment_type=carla.AttachmentType.Rigid)

    def __init__(self, player_actor,
            intersection_reader,
            scene_interval=32,
            record_interval=5,
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
        intersection_reader : IntersectionReader
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
        phi_attributes : attrdict.AttrDict
            Attributes T, T_past, B, A, ...etc to construct Phi object.
        """
        self.__player = player_actor
        self._intersection_reader = intersection_reader
        self.record_interval = record_interval
        self.scene_interval = scene_interval
        self.save_frequency = save_frequency
        self._save_directory = save_directory
        self.n_burn_frames = n_burn_frames
        self.episode = episode
        self.exclude_samples = exclude_samples
        self._debug = debug

        self.__make_sample_name = lambda frame : "{}/ep{:03d}/agent{:03d}/frame{:08d}".format(
                self._intersection_reader.map_name, self.episode, self.__player.id, frame)

        self.__world = self.__player.get_world()

        # __scene_builders : map of (int, SceneBuilder)
        #     Store scene builders by frame.
        self.__scene_builders = {}

        # __other_vehicles : list of carla.Vehicle
        #     List of IDs of vehicles not including __player.
        #     Use this to track other vehicles in the scene at each timestep. 
        self.__other_vehicles = None

        # __sensor : carla.Sensor
        #     Segmentation sensor. Data points will be used to construct overhead.
        self.__sensor = self.__create_segmentation_lidar_sensor()

        # __first_frame : int
        #     First frame in simulation. Used to find current timestep.
        self.__first_frame = None

    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: DataCollector.parse_image(weak_self, image))
    
    def stop_sensor(self):
        """Stop the sensor."""
        self.sensor.stop()

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.sensor.destroy()
        self.sensor = None

    def remove_scene_builder(self, frame)
        del self.__scene_builders[frame]

    def set_vehicles(self, vehicle_ids):
        """Given a list of non-player vehicle IDs retreive the vehicles corr.
        those IDs to watch.
        Used at the start of data collection. Should only be called once.
        Do not add the player vehicle ID in the list!
        """
        if self.__other_vehicles is None:
            vehicles = self.__world.get_actors(vehicle_ids)
            self.__other_vehicles = dict(zip(vehicle_ids, vehicles))
        else:
            raise Exception("Should not call more than once.")
    
    def capture_step(self, frame):
        if self.__first_frame is None:
            self.__first_frame = frame
        if frame - self.__first_frame < self.n_burn_frames:
            return
        if frame % self.record_interval == 0:
            scene_frame = frame % self.record_interval

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
        logging.debug(f"in LidarManager.parse_image() player = {self.__player.id} frame = {image.frame}")
        for scene_builder in self.__scene_builders.values():
            scene_builder.capture_lidar(image)


class OldDataCollector(object):
    """Data collector based on DIM."""

    def __create_lidar_sensor(self):
        return self._world.spawn_actor(
                create_lidar_blueprint_v2(self._world),
                carla.Transform(carla.Location(z=2.5)),
                attach_to=self._player,
                attachment_type=carla.AttachmentType.Rigid)
    
    def __create_segmentation_lidar_sensor(self):
        return self._world.spawn_actor(
                create_semantic_lidar_blueprint(self._world),
                carla.Transform(carla.Location(z=2.5)),
                attach_to=self._player,
                attachment_type=carla.AttachmentType.Rigid)

    def __init__(self, player_actor,
            intersection_reader,
            save_frequency=10,
            save_directory='out',
            n_burn_frames=60,
            episode=0,
            exclude_samples=SampleLabelFilter(),
            phi_attributes=DEFAULT_PHI_ATTRIBUTES,
            should_augment=False,
            n_augments=1,
            debug=False):
        """
        Parameters
        ----------
        player_actor : carla.Vehicle
            Vehicle that the data collector is following around, and should collect data for.
        intersection_reader : IntersectionReader
            One IntersectionReader instance should be shared by all data collectors.
        save_frequency : int
            Frequency (wrt. to CARLA simulator frame) to save data.
        n_burn_frames : int
            The number of initial frames to skip before saving data.
        episode : int
            Index of current episode to collect data from.
        exclude_samples : SampleLabelFilter
            Filter to exclude saving samples by label.
        phi_attributes : attrdict.AttrDict
            Attributes T, T_past, B, A, ...etc to construct Phi object.
        """
        self._player = player_actor
        self._intersection_reader = intersection_reader
        self.save_frequency = save_frequency
        self._save_directory = save_directory
        self.n_burn_frames = n_burn_frames
        self.episode = episode
        self.exclude_samples = exclude_samples
        self._debug = debug
        self.lidar_params = LidarParams()
        self._phi = create_phi(phi_attributes)
        _, _, self.T_past, _ = tensoru.shape(self._phi.S_past_world_frame)
        self.B, self.A, self.T, self.D = tensoru.shape(self._phi.S_future_world_frame)
        self.B, self.H, self.W, self.C = tensoru.shape(self._phi.overhead_features)
        self._make_sample_name = lambda frame : "{}/ep{:03d}/agent{:03d}/frame{:08d}".format(
                self._intersection_reader.map_name, self.episode, self._player.id, frame)
        self._world = self._player.get_world()
        self._other_vehicles = list()
        self._trajectory_size = max(self.T, self.T_past) + 1
        # player_transforms : collections.deque of carla.Transform
        self.player_transforms = collections.deque(
                maxlen=self._trajectory_size)
        # others_transforms : collections.deque of (dict of int : carla.Transform)
        #     Container of dict where key is other vehicle ID and value is
        #     carla.Transform
        self.others_transforms = collections.deque(
                maxlen=self._trajectory_size)
        self.trajectory_feeds = collections.OrderedDict()
        # lidar_feeds collections.OrderedDict
        #     Where int key is frame index and value
        #     is a carla.LidarMeasurement or carla.SemanticLidarMeasurement
        self.lidar_feeds = collections.OrderedDict()
        # __n_feeds : int
        #     Size of trajectory/lidar feed dict
        self.__n_feeds = self.T + 1
        self.streaming_generator = generate_observation.StreamingGenerator(
                self._phi, should_augment=should_augment, n_augments=n_augments)
        self.sensor = self.__create_segmentation_lidar_sensor()
        self._first_frame = None
    
    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: DataCollector._parse_image(weak_self, image))

    def stop_sensor(self):
        """Stop the sensor."""
        self.sensor.stop()

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.sensor.destroy()
        self.sensor = None
    
    def get_player(self):
        return self._player

    def set_vehicles(self, vehicle_ids):
        """Given a list of non-player vehicle IDs retreive the vehicles corr.
        those IDs to watch.
        Used at the start of data collection.
        Do not add the player vehicle ID in the list!
        """
        self._other_vehicles = self._world.get_actors(vehicle_ids)

    def _update_transforms(self):
        """Store player an other vehicle trajectories."""
        self.player_transforms.append(self._player.get_transform())
        others_transform = {}
        for vehicle in self._other_vehicles:
            others_transform[vehicle.id] = vehicle.get_transform()
        self.others_transforms.append(others_transform)

    def _should_save_dataset_sample(self, frame):
        """Check if collector reached a frame where it should save dataset sample.

        Parameters
        ----------
        frame : int
        """

        if len(self.trajectory_feeds) == 0:
            return False
        if frame - next(iter(self.trajectory_feeds)) > self.T:
            """Make sure that we can access past trajectories T steps
            ago relative to current frame."""
            if frame % self.save_frequency == 0:
                """Save dataset every save_frequency steps."""
                if frame - self._first_frame > self.n_burn_frames:
                    """Skip the first number of n_burn_frames"""
                    return True
        return False
    
    def _get_sample_labels(self):
        """Get labels for sample collected based on the sensor's current position. 

        Returns
        -------
        SampleLabelMap
        """
        intersection_type_label = self._intersection_reader \
                .at_intersection_to_label(self._player)
        slope_type_label, slope_pitch = self._intersection_reader \
                .at_slope_to_label(self._player)
        bounding_type_label = self._intersection_reader \
                .at_bounding_box_to_label(self._player)
        return SampleLabelMap(
                intersection_type=intersection_type_label,
                slope_type=slope_type_label,
                bounding_type=bounding_type_label,
                slope_pitch=slope_pitch)

    def debug_draw_red_player_bbox(self):
        self._world.debug.draw_box(
                carla.BoundingBox(
                    self._player.get_transform().location,
                    self._player.bounding_box.extent),
                self._player.get_transform().rotation,
                thickness=0.5,
                color=carla.Color(r=255, g=0, b=0, a=255),
                life_time=3.0)

    def debug_draw_green_player_bbox(self):
        self._world.debug.draw_box(
                carla.BoundingBox(
                        self._player.get_transform().location,
                        self._player.bounding_box.extent),
                self._player.get_transform().rotation,
                thickness=0.5,
                color=carla.Color(r=0, g=255, b=0, a=255),
                life_time=3.0)

    def _should_exclude_dataset_sample(self, sample_labels):
        """Check if collector should exclude saving sample at this sensor location.

        Parameters
        ----------
        sample_labels : SampleLabelMap

        Returns
        -------
        bool
        """
        for key, val in vars(sample_labels).items():
            if self.exclude_samples.contains(key, val):
                if self._debug:
                    self.debug_draw_red_player_bbox()
                    logging.debug("filter dataset sample")
                return True
        
        if self._debug:
            self.debug_draw_green_player_bbox()
        return False

    def capture_step(self, frame):
        """Have the data collector capture the current snapshot of the simulation.
        
        Parameters
        ----------
        frame : int
            The frame index returned from the latest call to carla.World.tick()
        """
        logging.debug(f"in LidarManager.capture_step() player = {self._player.id} frame = {frame}")
        if self._first_frame is None:
            self._first_frame = frame
        self._update_transforms()
        if len(self.player_transforms) >= self.T_past:
            """Only save trajectory feeds when we have collected at
            least T_past number of player and other vehicle transforms."""
            observation = generate_observation.PlayerObservation(
                    frame, self._phi, self._world, self._other_vehicles,
                    self.player_transforms, self.others_transforms,
                    self._player.bounding_box)
            self.streaming_generator.add_feed(
                    frame, observation, self.trajectory_feeds)
            
            if self._should_save_dataset_sample(frame):
                """Save dataset sample if needed."""
                logging.debug(f"saving sample. player = {self._player.id} frame = {frame}")
                sample_labels = self._get_sample_labels()
                if not self._should_exclude_dataset_sample(sample_labels):
                    self.streaming_generator.save_dataset_sample(
                            frame, self.episode, observation,
                            self.trajectory_feeds, self.lidar_feeds,
                            self._player.bounding_box,
                            self.sensor, self.lidar_params,
                            self._save_directory, self._make_sample_name,
                            sample_labels)
        
        if len(self.trajectory_feeds) > self.__n_feeds:
            """Remove older frames.
            (frame, feed) is removed in LIFO order."""
            frame, feed = self.trajectory_feeds.popitem(last=False)
            self.lidar_feeds.pop(frame)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        logging.debug(f"in LidarManager._parse_image() player = {self._player.id} frame = {image.frame}")
        self.lidar_feeds[image.frame] = image
