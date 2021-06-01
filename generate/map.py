from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import carla

import utility as util
import carlautil
import carlautil.debug

from .label import ScenarioIntersectionLabel, ScenarioSlopeLabel, BoundingRegionLabel
from .label import SampleLabelMap, SampleLabelFilter
from .label import SegmentationLabel

class MapData(object):
    def __init__(self, road_polygons, yellow_lines, white_lines):
        self.road_polygons = road_polygons
        self.yellow_lines = yellow_lines
        self.white_lines = white_lines

class MapQuerier(ABC):
    """Abstract class to keep track of properties in a map and
    used to query whether an actor is in a certain part
    (i.e. intersection, hill) of the map for the sake of
    labeling samples."""
    PRECISION = 0.05

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

    def __extract_polygons_and_lines(self):
        road_polygons = []
        yellow_lines = []
        white_lines = []
        topology      = [x[0] for x in self.carla_map.get_topology()]
        topology      = sorted(topology, key=lambda w: w.transform.location.z)
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(self.PRECISION)[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(self.PRECISION)[0]

            left_marking  = carlautil.locations_to_ndarray(
                    [self.__lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints])
            right_marking = carlautil.locations_to_ndarray(
                    [self.__lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints])
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
        return MapData(road_polygons, yellow_lines, white_lines)

    def __init__(self, carla_world, carla_map, debug=False):
        self.carla_world = carla_world
        self.carla_map = carla_map
        self.__debug = debug
        self.map_data = self.__extract_polygons_and_lines()

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

        if self.__debug:
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
