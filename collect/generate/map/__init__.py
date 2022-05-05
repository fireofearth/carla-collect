import os
from abc import ABC, abstractmethod
import logging

import dill
import shapely
import shapely.geometry
import networkx as nx
import numpy as np
import pandas as pd
import carla

import utility as util
import utility.shu
import carlautil
import carlautil.debug

from ..label import (
    ScenarioIntersectionLabel, ScenarioSlopeLabel, BoundingRegionLabel,
    SampleLabelMap, SampleLabelFilter, SegmentationLabel
)
from .road import (
    get_road_segment_enclosure,
    cover_along_waypoints_fixedsize,
    RoadBoundaryConstraint
)
from ...visualize.trajectron import render_entire_map, render_map_crop
from ... import CACHEDIR

logger = logging.getLogger(__name__)
CARLA_MAP_NAMES = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]

class MapData(object):
    """Map data.
    TODO: DEPRECATED

    Attributes
    ==========
    road_polygons : list of np.array
        List of road polygons (closed), each represented as array of shape (?, 2)
    yellow_lines : list of  np.array
        List of white road lines, each represented as array of shape (?, 2)
    white_lines : list of np.array
        List of white road lines, each represented as array of shape (?, 2)
    """
    def __init__(self, road_polygons, yellow_lines, white_lines):
        self.road_polygons = road_polygons
        self.yellow_lines = yellow_lines
        self.white_lines = white_lines


class MapDataExtractor(object):
    """Utility class to extract properties of the CARLA map."""

    def __init__(self, carla_world, carla_map):
        """Constructor.
        
        Parameters
        ==========
        carla_world: carla.World
            CARLA World.
        carla_map: carla.Map
            CARLA Map.
        """
        self.carla_world = carla_world
        self.carla_map = carla_map
    
    @staticmethod
    def __lateral_shift(transform, shift):
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def __is_yellow_line(self, waypoint, shift):
        w = self.carla_map.get_waypoint(self.__lateral_shift(waypoint.transform, shift),
                project_to_road=False)
        if w is None:
            return False
        return w.lane_id * waypoint.lane_id < 0

    def extract_road_polygons_and_lines(self, sampling_precision=0.05):
        """Extract road white and yellow dividing lines, and road polygons.

        Parameters
        ==========
        sampling_precision : float
            Space between sampling points to create road data.

        Returns
        =======
        util.AttrDict
            Payload of road network information with the following keys:

            * road_polygons: list of road polygons (closed), each represented as array of shape (?, 2)
            * yellow_lines: list of white road lines, each represented as array of shape (?, 2)
            * white_lines: list of white road lines, each represented as array of shape (?, 2)
        """
        road_polygons = []
        yellow_lines  = []
        white_lines   = []
        topology      = [x[0] for x in self.carla_map.get_topology()]
        topology      = sorted(topology, key=lambda w: w.transform.location.z)
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(sampling_precision)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(sampling_precision)[0]

            left_marking  = carlautil.locations_to_ndarray(
                    [self.__lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints],
                    flip_y=True)
            right_marking = carlautil.locations_to_ndarray(
                    [self.__lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints],
                    flip_y=True)
            road_polygon = np.concatenate((left_marking, np.flipud(right_marking)), axis=0)

            if len(road_polygon) > 2:
                road_polygons.append(road_polygon)
                if not waypoint.is_intersection:
                    sample = waypoints[int(len(waypoints) / 2)]
                    if self.__is_yellow_line(sample, -sample.lane_width * 1.1):
                        yellow_lines.append(left_marking)
                    else:
                        white_lines.append(left_marking)
                    if self.__is_yellow_line(sample, sample.lane_width * 1.1):
                        yellow_lines.append(right_marking)
                    else:
                        white_lines.append(right_marking)

        return util.AttrDict(road_polygons=road_polygons,
                yellow_lines=yellow_lines, white_lines=white_lines)

    def extract_waypoint_points(self, sampling_precision=4.0, slope_degrees=5.0):
        """Extract slope topology of the map"""
        slope_udegrees = (360. - slope_degrees)
        wps = self.carla_map.generate_waypoints(sampling_precision)
        def h(wp):
            l = wp.transform.location
            pitch = wp.transform.rotation.pitch
            return np.array([l.x, l.y, l.z, pitch])
        loc_and_pitch_of_wps = util.map_to_ndarray(h, wps)
        wp_locations = loc_and_pitch_of_wps[:, :-1]
        wp_pitches = loc_and_pitch_of_wps[:, -1]
        self.wp_pitches = self.wp_pitches % 360.
        wp_is_sloped = np.logical_and(
                slope_degrees < self.wp_pitches,
                self.wp_pitches < slope_udegrees)
        return pd.DataFrame({'x': wp_locations[0], 'y': wp_locations[1], 'z': wp_locations[2],
                'pitch': wp_pitches, 'is_sloped': wp_is_sloped})
    
    def extract_spawn_points(self):
        spawn_points = self.carla_map.get_spawn_points()
        return np.concatenate((
            carlautil.to_locations_ndarray(spawn_points, flip_y=True),
            carlautil.to_rotations_ndarray(spawn_points, flip_y=True)), axis=-1)

    def extract_junction_with_portals(self):
        """Extract junctions from the map 

        Returns
        =======
        list of util.AttrDict
            Each of junction has these attributes:

            * **pos** : ndarray
              - The (x, y) position of the junction.
            * **waypoints** : ndarray
              - Pairs of waypoint entering and exiting the junction f shape (n pairs, 2, 4).
              Each waypoint is (x position, y position, yaw, road length).

        """
        carla_topology = self.carla_map.get_topology()
        junctions = carlautil.get_junctions_from_topology_graph(carla_topology)
        _junctions = []
        for junction in junctions:
            jx, jy, _ = carlautil.to_location_ndarray(junction, flip_y=True)
            wps = junction.get_waypoints(carla.LaneType.Driving)
            _wps = []
            for wp1, wp2 in wps:
                # wp1 is the waypoint entering into the intersection
                # wp2 is the waypoint exiting out of the intersection
                x, y, _ = carlautil.to_location_ndarray(wp1, flip_y=True)
                _, yaw, _ = carlautil.to_rotation_ndarray(wp1, flip_y=True)
                _wp1 = (x, y, yaw, wp1.lane_width)
                x, y, _ = carlautil.to_location_ndarray(wp2, flip_y=True)
                _, yaw, _ = carlautil.to_rotation_ndarray(wp2, flip_y=True)
                _wp2 = (x, y, yaw, wp2.lane_width)
                _wps.append((_wp1, _wp2))
            _junctions.append(util.AttrDict(pos=np.array([jx, jy]), waypoints=np.array(_wps)))

        return _junctions

    def extract_junction_points(self, tlight_find_radius=25.):
        carla_topology = self.carla_map.get_topology()
        junctions = carlautil.get_junctions_from_topology_graph(carla_topology)
        tlights = util.filter_to_list(lambda a: 'traffic_light' in a.type_id,
                self.carla_world.get_actors())
        tlight_distances = np.zeros((len(tlights), len(junctions),))
        f = lambda j: carlautil.location_to_ndarray(j.bounding_box.location)
        junction_locations = util.map_to_ndarray(f, junctions)
        
        g = lambda tl: carlautil.transform_to_location_ndarray(
                tl.get_transform())
        tlight_locations = util.map_to_ndarray(g, tlights)

        for idx, junction in enumerate(junctions):
            tlight_distances[:,idx] = np.linalg.norm(
                    tlight_locations - junction_locations[idx], axis=1)

        is_controlled_junction = (tlight_distances < tlight_find_radius).any(axis=0)
        is_uncontrolled_junction = np.logical_not(is_controlled_junction)
        controlled_junction_locations \
                = junction_locations[is_controlled_junction]
        uncontrolled_junction_locations \
                = junction_locations[is_uncontrolled_junction]
        return util.AttrDict(
                controlled=controlled_junction_locations,
                uncontrolled=uncontrolled_junction_locations)


class CachedMapData(object):
    """Manages the persisting of map data from MapDataExtractor
    to cache in a way that can be loaded for other purposes.
    
    Attributes
    ==========
    map_datum : dict of (str, util.AttrDict)
        Name of map to road data consisting of:
        road_polygons, white_lines, yellow_lines, junctions, spawn_points.
        Road polygons and lines are obtained from
        :meth:`~collect.generate.map.MapDataExtractor.extract_road_polygons_and_lines`.
        Junctions are obtained from
        :meth:`~collect.generate.map.MapDataExtractor.extract_junction_with_portals`.
        Spawn points are obtained from 
        :meth:`~collect.generate.map.MapDataExtractor.extract_spawn_points`.
    map_to_smpolys : dict of (str, (list of MultiPolygon))
        Name of map to Shapely MultiPolygon boxes bounding junction entrance/exits.
    map_to_scircles : dict of (str, (list of Polygon))
        Name of map to Shapely circle covering junction region.
    """

    TLIGHT_DETECT_RADIUS = 25.0

    @staticmethod
    def save_map_data_to_cache(client):
        """Extract data from all maps through a CARLA client and save to disk. 

        Parameters
        ==========
        client : carla.Client
            Client to access live map data.
        """
        carla_world = client.get_world()
        carla_map = carla_world.get_map()
        for map_name in CARLA_MAP_NAMES:
            logger.info(f"Caching map data from {map_name}.")
            if carla_map.name != map_name:
                carla_world = client.load_world(map_name)
                carla_map = carla_world.get_map()
            extractor = MapDataExtractor(carla_world, carla_map)
            logger.info("    Extracting and caching road polygons and dividers")
            p = extractor.extract_road_polygons_and_lines()
            road_polygons, yellow_lines, white_lines = (
                p.road_polygons,
                p.yellow_lines,
                p.white_lines,
            )
            logger.info("    Extracting and caching road junctions")
            junctions = extractor.extract_junction_with_portals()
            logger.info("    Extracting and caching spawn points")
            spawn_points = extractor.extract_spawn_points()
            payload = {
                "road_polygons": road_polygons,
                "yellow_lines": yellow_lines,
                "white_lines": white_lines,
                "junctions": junctions,
                "spawn_points": spawn_points,
            }
            os.makedirs(CACHEDIR, exist_ok=True)
            cachepath = f"{CACHEDIR}/map_data.{map_name}.pkl"
            with open(cachepath, "wb") as f:
                dill.dump(payload, f, protocol=dill.HIGHEST_PROTOCOL)
        logger.info("Done.")

    def __init__(self):
        """Load map data from cache and collect shapes of all the intersections."""
        self.map_datum = { }
        self.map_to_smpolys = {}
        self.map_to_scircles = {}
        
        logger.info("Retrieve map from cache.")
        for map_name in CARLA_MAP_NAMES:
            cachepath = f"{CACHEDIR}/map_data.{map_name}.pkl"
            with open(cachepath, "rb") as f:
                payload = dill.load(f, encoding="latin1")
            self.map_datum[map_name] = util.AttrDict(
                road_polygons=payload["road_polygons"],
                white_lines=payload["white_lines"],
                yellow_lines=payload["yellow_lines"],
                junctions=payload["junctions"],
                spawn_points=payload["spawn_points"]
            )

        logger.info("Retrieving some data from map.")
        for map_name in CARLA_MAP_NAMES:
            for _junction in self.map_datum[map_name].junctions:
                f = lambda x, y, yaw, l: util.vertices_from_bbox(
                    np.array([x, y]), yaw, np.array([5.0, 0.95 * l])
                )
                vertex_set = util.map_to_ndarray(
                    lambda wps: util.starmap(f, wps), _junction.waypoints
                )
                smpolys = util.map_to_list(util.shu.vertex_set_to_smpoly, vertex_set)
                util.setget_list_from_dict(self.map_to_smpolys, map_name).extend(smpolys)
                x, y = _junction.pos
                scircle = shapely.geometry.Point(x, y).buffer(self.TLIGHT_DETECT_RADIUS)
                util.setget_list_from_dict(self.map_to_scircles, map_name).append(scircle)


class MapQuerier(ABC):
    """Abstract class to keep track of properties in a map and
    used to query whether an actor is in a certain part
    (i.e. intersection, hill) of the map for the sake of
    labeling samples."""

    # used by __extract_polygons_and_lines()
    EXTRACT_PRECISION = 0.05

    def __init__(self, carla_world, carla_map, debug=False):
        self.carla_world = carla_world
        self.carla_map = carla_map
        self._debug = debug
        self.map_data_extractor = MapDataExtractor(self.carla_world, self.carla_map)
        self.map_data = self.map_data_extractor.extract_road_polygons_and_lines(
                sampling_precision=self.EXTRACT_PRECISION)

    @property
    def map_name(self):
        return self.carla_map.name.split('/')[-1]

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

    def road_segment_enclosure_from_actor(self, actor, tol=2.0):
        """Get box enclosure from straight road.
        
        TODO: data should be flipped about x-axis
        """
        t = actor.get_transform()
        wp = self.carla_map.get_waypoint(t.location)
        return get_road_segment_enclosure(wp, tol=tol)
    
    def curved_road_segments_enclosure_from_actor(
        self, actor: carla.Actor, max_distance: float, choices=[], flip_x=False, flip_y=False
    ) -> util.AttrDict:
        """Get segmented enclosure of fixed size from curved road.

        Parameters
        ==========
        actor : carla.Actor
            Position of actor to get starting waypoint.
        max_distance : float
            The max distance of the path starting from `start_wp`. Use to specify length of path.
        choices : list of int
            The indices of the turns to make when reaching a fork on the road network.
            `choices[0]` is the index of the first turn, `choices[1]` is the index of the second turn, etc.

        Returns
        =======
        util.AttrDict
            Container of road segment properties:

            * **spline** : scipy.interpolate.CubicSpline
              - The spline representing the path the vehicle should motion plan on.
            * **polytopes** : list of (ndarray, ndarray)
              - List of polytopes in H-representation (A, b)
              where x is in polytope if Ax <= b.
            * **distances** : ndarray
              - The distances along the spline to follow from nearest endpoint
              before encountering corresponding covering polytope in index.
            * **positions** : ndarray
              - The 2D positions of center of the covering polytope in index.
        """
        t = actor.get_transform()
        wp = self.carla_map.get_waypoint(t.location)
        wp = wp.previous(5)[0]
        lane_width = wp.lane_width
        return cover_along_waypoints_fixedsize(
            wp, choices, max_distance + 7, lane_width, flip_x=flip_x, flip_y=flip_y
        )

    def road_boundary_constraints_from_actor(
        self, actor: carla.Actor, max_distance: float, choices=[], flip_x=False, flip_y=False
    ) -> RoadBoundaryConstraint:
        """Get segmented enclosure of fixed size from curved road.

        Parameters
        ==========
        actor : carla.Actor
            Position of actor to get starting waypoint.
        max_distance : float
            The max distance of the path starting from `start_wp`. Use to specify length of path.
        choices : list of int
            The indices of the turns to make when reaching a fork on the road network.
            `choices[0]` is the index of the first turn, `choices[1]` is the index of the second turn, etc.
        
        Returns
        =======
        RoadBoundaryConstraint
            The road boundary constraints.
        
        TODO: MapQuerier.road_segment_enclosure_from_actor() and
        MapQuerier.curved_road_segments_enclosure_from_actor()
        should be refactored to use RoadBoundaryConstraint as a factory.
        """
        t = actor.get_transform()
        wp = self.carla_map.get_waypoint(t.location)
        wp = wp.previous(5)[0]
        lane_width = wp.lane_width
        return RoadBoundaryConstraint(
            wp, max_distance + 7, lane_width, choices, flip_x=flip_x, flip_y=flip_y
        )

    def render_map(self, ax, extent=None):
        """Render the map.

        Parameters
        ==========
        ax : matplotlib.axes.Axes
        extent : tuple of int
            The extent of the map to render of form (x_min, x_max, y_min, y_max) in meters.
            If not passed, then render entire map.
        """
        if extent is None:
            render_entire_map(ax, self.map_data)
        else:
            render_map_crop(ax, self.map_data, extent)


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

    TODO: refactor to use MapDataExtractor to get map data.
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
