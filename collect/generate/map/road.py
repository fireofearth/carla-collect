"""
"""

import collections
import functools
import math

import numpy as np
import pandas as pd
import scipy
import scipy.interpolate

import carla
import utility as util
import utility.npu
import carlautil

from ...exception import CollectException

class RoadException(CollectException):
    pass

PRECISION = 1.0

def to_point(wp, flip_x=False, flip_y=False):
    """Convert waypoint to a 2D point.

    Parameters
    ==========
    wp : carla.Waypoint

    Returns
    =======
    list of float
    """
    x, y, _ = carlautil.to_location_ndarray(wp, flip_x=flip_x, flip_y=flip_y)
    return [x, y]

def get_adjacent_waypoints(start_wp):
    """Get the lane waypoints adjacent to the waypoint going in the same direction on the road.
    
    Parameters
    ==========
    start_wp : carla.Waypoint
        Starting waypoint to get adjacent lane waypoints from.
    
    Returns
    =======
    list of carla.Waypoint
        Waypoints of form [--left lanes--, start_wp, --right lanes--]
        ordered from left-most waypoint to rightmost waypoint.
    """
    lane_sgn = util.sgn(start_wp.lane_id)
    def should_select(wp):
        if wp is None:
            return False
        if wp.lane_type != carla.LaneType.Driving:
            return False
        if util.sgn(wp.lane_id) != lane_sgn:
            return False
        return True
    rlanes = list()
    wp = start_wp
    while True:
        wp = wp.get_right_lane()
        if not should_select(wp):
            break
        rlanes.append(wp)
    llanes = list()
    wp = start_wp
    while True:
        wp = wp.get_left_lane()
        if not should_select(wp):
            break
        llanes.append(wp)
    return util.reverse_list(llanes) + [start_wp] + rlanes

def get_straight_line(start_wp, start_yaw, tol=2.0):
    """Get the points corresponding to the straight part of the road lane starting from start_wp.

    Parameters
    ==========
    start_wp : carla.Waypoint
        The starting point of the road lane.
    start_yaw : float
        The angle of the road in radians.
        This is passed as the lanes on the road may not be parallel.
    tol : float (optional)
        The tolerance to select straight part of the road in meters.

    Returns
    =======
    np.array
        Array of points of shape (N, 2).
        Each (x, y) point is in meters in world coordinates.
    """
    x_start, y_start, _ = carlautil.to_location_ndarray(start_wp)
    FACTOR = 10
    x_end, y_end = x_start + FACTOR * np.cos(start_yaw), y_start + FACTOR * np.sin(start_yaw)
    
    def inner(wp):
        wps = wp.next_until_lane_end(PRECISION)
        points = np.array(util.map_to_ndarray(to_point, wps))
        distances = util.distances_from_line_2d(points, x_start, y_start, x_end, y_end)
        wps_mask = np.nonzero(distances > tol)[0]
        if wps_mask.size > 0:
            # waypoints veer off a straight line
            idx = wps_mask[0]
            return points[:idx], list()
        else:
            # waypoints stay one straight line. Get the next set of waypoints
            return points, wps[-1].next(PRECISION)
    
    dq = collections.deque([start_wp])
    point_collection = []
    while len(dq) > 0:
        wp = dq.popleft()
        points, next_wps = inner(wp)
        dq.extend(next_wps)
        point_collection.append(points)
    return np.concatenate(point_collection)

def get_straight_lanes(start_wp, tol=2.0):
    """Get the points corresponding to the straight parts of all
    lanes in road starting from start_wp going in the same direction.

    Parameters
    ==========
    start_wp : carla.Waypoint
        The starting point of the road lane.
    tol : float (optional)
        The tolerance to select straight part of the road in meters.

    Returns
    =======
    list of np.array
        Each item in the list are points for one lane of of shape (N_i, 2).
        Each (x, y) point is in meters in world coordinates.
    """
    _, start_yaw, _ = carlautil.to_rotation_ndarray(start_wp)
    wps = get_adjacent_waypoints(start_wp)
    f = lambda wp: get_straight_line(wp, start_yaw, tol=tol)
    return np.concatenate(util.map_to_list(f, wps))

def get_road_segment_enclosure(start_wp, tol=2.0):
    """Get rectangle that tightly inner approximates of the road segment
    containing the starting waypoint.

    Parameters
    ==========
    start_wp : carla.Waypoint
        A starting waypoint of the road.
    tol : float (optional)
        The tolerance to select straight part of the road in meters.

    Returns
    =======
    np.array
        The position and the heading angle of the starting waypoint
        of the road of form [x, y, angle] in (meters, meters, radians).
    np.array
        The 2D bounding box enclosure in world coordinates of shape (4, 2)
        enclosing the road segment.
    np.array
        The parameters of the enclosure of form
        (b_length, f_length, r_width, l_width)
        If the enclosure is in the reference frame such that the starting
        waypoint points along +x-axis, then the enclosure has these length
        and widths:
        ____________________________________
        |              l_width             |
        |               |                  |
        |               |                  |
        | b_length -- (x, y)-> -- f_length |
        |               |                  |
        |               |                  |
        |              r_width             |
        ------------------------------------
    """
    _LENGTH = -2
    _, start_yaw, _ = carlautil.to_rotation_ndarray(start_wp)
    adj_wps = get_adjacent_waypoints(start_wp)
    # mtx : np.array
    #   Rotation matrix from world coordinates, both frames in UE orientation
    mtx = util.rotation_2d(-start_yaw)
    # rev_mtx : np.array
    #   Rotation matrix to world coordinates, both frames in UE orientation
    rev_mtx = util.rotation_2d(start_yaw)
    s_x, s_y, _ = carlautil.to_location_ndarray(start_wp)

    # Get points of lanes
    f = lambda wp: get_straight_line(wp, start_yaw, tol=tol)
    pc = util.map_to_list(f, adj_wps)
    
    # Get length of bb for lanes
    def g(points):
        points = points - np.array([s_x, s_y])
        points = (rev_mtx @ points.T)[0]
        return np.abs(np.max(points) - np.min(points))
    lane_lengths = util.map_to_ndarray(g, pc)
    length = np.min(lane_lengths)

    # Get width of bb for lanes
    lwp, rwp = adj_wps[0], adj_wps[-1]
    l_x, l_y, _ = carlautil.to_location_ndarray(lwp)
    r_x, r_y, _ = carlautil.to_location_ndarray(rwp)
    points = np.array([[l_x, l_y], [s_x, s_y], [r_x, r_y]])
    points = points @ rev_mtx.T
    l_width = np.abs(points[0, 1] - points[1, 1]) + lwp.lane_width / 2.
    r_width = np.abs(points[1, 1] - points[2, 1]) + rwp.lane_width / 2.

    # construct bounding box of road segment
    x, y, _ = carlautil.to_location_ndarray(start_wp)
    vec = np.array([[0,0], [_LENGTH, 0]]) @ mtx.T
    dx0, dy0 = vec[1, 0], vec[1, 1]
    vec = np.array([[0,0], [length, 0]]) @ mtx.T
    dx1, dy1 = vec[1, 0], vec[1, 1]
    vec = np.array([[0,0], [0, -l_width]]) @ mtx.T
    dx2, dy2 = vec[1, 0], vec[1, 1]
    vec = np.array([[0,0], [0, r_width]]) @ mtx.T
    dx3, dy3 = vec[1, 0], vec[1, 1]
    bbox = np.array([
            [x + dx0 + dx3, y + dy0 + dy3],
            [x + dx1 + dx3, y + dy1 + dy3],
            [x + dx1 + dx2, y + dy1 + dy2],
            [x + dx0 + dx2, y + dy0 + dy2]])
    start_wp_spec = np.array([s_x, s_y, start_yaw])
    bbox_spec = np.array([_LENGTH, length, r_width, l_width])
    return start_wp_spec, bbox, bbox_spec

def split_line_by_mask(X, mask):
    # split X by inclusion or exclusion
    indices = np.where(np.diff(mask,prepend=np.nan))[0]
    l = np.split(X, indices)[1:]
    X_inclusion = l[::2] if mask[0] else l[1::2]
    X_exclusion = l[::2] if not mask[0] else l[1::2]
    return X_inclusion, X_exclusion

def remove_line_segments_by_condition(cond, lines):
    _lines = []
    for line in lines:
        splits, _ = split_line_by_mask(line, cond(line))
        _lines += splits
    return _lines

def split_polygon_by_mask(X, mask):
    # split X by inclusion or exclusion
    indices = np.where(np.diff(mask,prepend=np.nan))[0]
    l = np.split(X, indices)[1:]
    X_inclusion = l[::2] if mask[0] else l[1::2]
    X_exclusion = l[::2] if not mask[0] else l[1::2]
    if len(X_inclusion) % 2 == 0:
        # fold all
        n = len(X_inclusion)
        X_inclusion = util.map_to_list(np.concatenate, zip(X_inclusion[:n//2], X_inclusion[:n//2 - 1:-1]))
        # take middle and fold rest
        n = len(X_exclusion)
        X_exclusion = util.map_to_list(np.concatenate, zip(X_exclusion[:n // 2], X_exclusion[:n // 2:-1])) \
                + [X_exclusion[n // 2]]
    else:
        # take middle and fold rest
        n = len(X_inclusion)
        X_inclusion = util.map_to_list(np.concatenate, zip(X_inclusion[:n // 2], X_inclusion[:n // 2:-1])) \
                + [X_inclusion[n // 2]]
        # fold all
        n = len(X_exclusion)
        X_exclusion = util.map_to_list(np.concatenate, zip(X_exclusion[:n//2], X_exclusion[:n//2 - 1:-1]))
    return X_inclusion, X_exclusion

def remove_polygons_by_condition(cond, polygons):
    _polygons = []
    for polygon in polygons:
        splits, _ = split_polygon_by_mask(polygon, cond(polygon))
        _polygons += splits
    return _polygons

######################################################################
# Create fixed size covering polytopes over a path on the road network
######################################################################

def compute_segment_length(delta, k):
    return 2*np.arccos(1/(k*delta + 1))/k

def cover_along_waypoints_fixedsize(start_wp, choices, max_distance, lane_width,
        flip_x=False, flip_y=False):
    """Compute covering polytopes over a path on the road network.

    Parameters
    ==========
    start_wp : carla.Waypoint
        Starting waypoint of the path on the road network.
    choices : list of int
        The indices of the turns to make when reaching a fork on the road network.
        `choices[0]` is the index of the first turn, `choices[1]` is the index of the second turn, etc.
    max_distance : float
        The max distance of the path starting from `start_wp`. Use to specify length of path.
    lane_width : float
        The width of the road.

    Returns
    =======
    util.AttrDict
        The data of the covering poytopes.
        - spline    : the spline fitting the path of road network.
        - max_k     : approx. max curvature of the spline.
        - segment_length : length of spline segment covered by polytope. 
        - polytopes : list of ndarray polytopes with H-representation (A, b)
                      where points Ax <= b iff x is in the polytope.
        - distances : ndarray of distances along the spline to follow from nearest
                      endpoint before encountering corresponding covering polytope
                      in index.
        - positions : ndarray of 2D positions of center of the covering polytope
                      in index.
        - tangents  : list of tuple of ndarray of ndarray
                      The tangent vector components of the tangents entering and
                      exiting the spline.
    """
    waypoints, points, distances = carlautil.collect_points_along_path(
        start_wp, choices, max_distance, flip_x=flip_x, flip_y=flip_y
    )
    # fit a spline and get the 1st and 2nd spline derivatives
    # distances = util.npu.cumulative_points_distances(points)
    # distances = np.insert(distances, 0, 0)
    L = distances[-1]
    spline = scipy.interpolate.CubicSpline(distances, points, axis=0)
    dspline = spline.derivative(1)
    ddspline = spline.derivative(2)

    # compute approx. max curvature
    # distances = np.linspace(0, L, 10)
    max_k = np.max(np.linalg.norm(ddspline(distances), axis=1))

    # compute the vertices of road covers
    half_lane_width = 0.55*lane_width
    segment_length = min(10, compute_segment_length(0.25, max_k))
    n = int(np.round(L / segment_length))
    distances = np.linspace(0, L, n)
    l = util.pairwise(zip(spline(distances), dspline(distances), ddspline(distances)))
    polytopes  = [] # polytope representation of rectangular cover
    vertex_set = [] # vertex representation of rectangular cover
    tangents   = []
    for (X1, dX1, ddX1), (X2, dX2, ddX2) in l:
        sgn1 = np.sign(dX1[0]*ddX1[1] - dX1[1]*ddX1[0])
        sgn2 = np.sign(dX2[0]*ddX2[1] - dX2[1]*ddX2[0])
        tangent1 = ddX1 / np.linalg.norm(ddX1)
        tangent2 = ddX2 / np.linalg.norm(ddX2)
        p1 = X1 + half_lane_width*sgn1*tangent1
        p2 = X2 + half_lane_width*sgn2*tangent2
        p3 = X2 - half_lane_width*sgn2*tangent2
        p4 = X1 - half_lane_width*sgn1*tangent1
        vertices = np.stack((p1, p2, p3, p4))
        A, b = util.npu.vertices_to_halfspace_representation(vertices)
        polytopes.append((A, b))
        vertex_set.append(vertices)
        tangents.append((dX1, dX2))
    return util.AttrDict(
        spline=spline,
        max_k=max_k,
        segment_length=segment_length,
        polytopes=polytopes,
        distances=distances,
        positions=np.mean(np.stack(vertex_set), axis=1),
        tangents=tangents
    )

########################################################################
# Create varying size covering polytopes over a path on the road network
########################################################################

class RoadBoundaryConstraint(object):
    """Create bounding boxes for road boundary.
    
    Attributes
    ==========
    road_segs: util.AttrDict
        Container of road segment properties.
        Output of cover_along_waypoints_varyingsize()
    """
    
    __DIFF = 0.5
    __STEPSIZE = 1.0
    __VIOLTOL = 0.03
    __PRECISION = 1.0
    __DELTA = 0.25
    __K = 3

    @staticmethod
    def __pad_junction_mask(mask):
        diff_mask = np.diff(mask)
        diff_mask = np.r_[[False], diff_mask] | np.r_[diff_mask, [False]]
        mask &= ~diff_mask
        return mask

    @staticmethod
    def __split_line_by_mask(X, mask):
        """Split lines of items into boundary
        overlapping segments according to mask."""
        indices = np.where(np.diff(mask, prepend=np.nan))[0][1:]
        splits = []
        for i in range(len(indices)):
            if i == 0:
                splits.append(X[:indices[i]+1])
            else:
                splits.append(X[indices[i-1]:indices[i]+1])
        splits.append(X[indices[-1]:])
        split_mask = (np.arange(len(splits)) % 2 == 0) ^ ~mask[0]
        return splits, split_mask
    
    @staticmethod
    def __interval_index_ids(wps, distances):
        data = util.map_to_ndarray(
            lambda wp: [wp.is_junction, wp.road_id, wp.section_id, wp.lane_id], wps
        )[1:]
        return pd.DataFrame(
            data,
            columns=["is_junction", "road_id", "section_id", "lane_id"],
            index=pd.IntervalIndex.from_breaks(distances)
        )

    def compute_curvature(self, ddspline, d):
        """Find the curvature given distance from start of spline."""
        return np.linalg.norm(ddspline(d))
    
    def compute_segment_length(self, k):
        """Compute the segment length for a curvature."""
        return 2*np.arccos(1/(k*self.__DELTA + 1))/k

    def compute_violation(self, s, k):
        """Compute the amount of violation in range [0, 1]
        from given segment of length s covering a road with curvature k"""
        if s*k < math.pi:
            return max(0., 1 - (1 + self.__DELTA*k)*np.cos(0.5*k*s))
        else:
            return 1.

    def __compute_cover_vertices_disc(self, spline, dspline, ddspline, dist1, dist2):
        """Compute vertices of 4 sided cover for spline.
        NOTE: doesn't use derivative to create cover"""
        X1    = spline(dist1)
        X2    = spline(dist2)
        X1p   = spline(dist1 + self.__DIFF)
        unit1 = util.npu.unit_normal_2d(X1, X1p)
        X2p   = spline(dist2 - self.__DIFF)
        unit2 = util.npu.unit_normal_2d(X2p, X2)
        p1 = X1 + self.delta*unit1
        p2 = X2 + self.delta*unit2
        p3 = X2 - self.delta*unit2
        p4 = X1 - self.delta*unit1
        return np.stack((p1, p2, p3, p4))

    def __compute_cover_vertices(self, spline, dspline, ddspline, dist1, dist2):
        """Compute vertices of 4 sided cover for spline.
        Slight/serious(?) loss of performance to when covering CARLA waypoints."""
        X1   = spline(dist1)
        dX1  = dspline(dist1)
        ddX1 = ddspline(dist1)
        X2   = spline(dist2)
        dX2  = dspline(dist2)
        ddX2 = ddspline(dist2)
        sgn1 = np.sign(dX1[0]*ddX1[1] - dX1[1]*ddX1[0])
        sgn2 = np.sign(dX2[0]*ddX2[1] - dX2[1]*ddX2[0])
        p1 = X1 + self.delta*sgn1* ddX1 / np.linalg.norm(ddX1)
        p2 = X2 + self.delta*sgn2* ddX2 / np.linalg.norm(ddX2)
        p3 = X2 - self.delta*sgn2* ddX2 / np.linalg.norm(ddX2)
        p4 = X1 - self.delta*sgn1* ddX1 / np.linalg.norm(ddX1)
        return np.stack((p1, p2, p3, p4))
    
    def cover_along_path_varyingsize(self):
        """Cover path with varying size polytopes.

        Returns
        =======
        util.AttrDict
            Container of road segment properties.
            - polytopes : list of (ndarray, ndarray)
                List of polytopes in H-representation (A, b)
                where x is in polytope if Ax <= b.
            - segment_lengths : list of float
                Lengths of spline segments covered by the polytope.
            - mask : ndarray of bool
                True if segment covers a junction, False otherwise.
            - distances : list of float
                The distances along the splines to follow from nearest endpoint
                before encountering corresponding covering polytope in index.
            - positions : ndarray
                The 2D positions of center of the covering polytope in index.
            - ids : pd.DataFrame
                The road data of each road segment.
                Has columns: polytope_id, is_junction, road_id, section_id, lane_id.
        """
        polytopes = []
        vertex_set = []
        mask = []
        segment_lengths = []
        postprocess_distances = [0.]
        ids = []
        for in_junction, distances, spline, dspline, ddspline in zip(
            self.split_mask, self.distances_splits, self.splines, self.dsplines, self.ddsplines
        ):
            acc_distance = distances[0]
            max_distance = distances[-1]
            # print(acc_distance, max_distance)
            while acc_distance < max_distance:
                k = self.compute_curvature(ddspline, acc_distance)
                segment_length = self.compute_segment_length(k)
                segment_length = min(segment_length, max_distance - acc_distance)
                query_distances = np.arange(acc_distance, acc_distance + segment_length, self.__STEPSIZE)
                query_distance = None
                if query_distances.size == 1:
                    """place bounding box between acc_distance and (acc_distance + segment_length)"""
                    next_distance = acc_distance + segment_length
                else:
                    """since curvature of spline may change, figure out the maximum distance ahead
                    from initial distance where the curvature does not change drastically."""
                    for query_distance in query_distances:
                        k1 = self.compute_curvature(ddspline, query_distance)
                        k2 = self.compute_curvature(ddspline, query_distance + self.__STEPSIZE)
                        has_viol1 = self.compute_violation(segment_length, k1) > self.__VIOLTOL
                        has_viol2 = self.compute_violation(segment_length, k2) > self.__VIOLTOL
                        if has_viol1 and has_viol2:
                            break
                    if query_distance == query_distances[-1]:
                        next_distance = acc_distance + segment_length
                    else:
                        next_distance = query_distance
                        segment_length = next_distance - acc_distance
                if acc_distance == next_distance:
                    raise Exception("no progression")
                vertices = self.__compute_cover_vertices_disc(
                    spline, dspline, ddspline, acc_distance, next_distance
                )
                A, b = util.npu.vertices_to_halfspace_representation(vertices)
                polytopes.append((A, b))
                vertex_set.append(vertices)
                mask.append(in_junction)
                segment_lengths.append(segment_length)
                postprocess_distances.append(next_distance)
                ids.append(self.ids_df.loc[next_distance - acc_distance].values)
                acc_distance = next_distance
        ids = pd.DataFrame(
            ids,
            columns=["is_junction", "road_id", "section_id", "lane_id"],
            index=pd.IntervalIndex.from_breaks(postprocess_distances)
        )
        ids["polytope_id"] = range(len(polytopes))
        return util.AttrDict(
            polytopes=polytopes,
            segment_lengths=segment_lengths,
            mask=np.array(mask),
            distances=postprocess_distances,
            positions=np.mean(np.stack(vertex_set), axis=1),
            ids=ids
        )
    
    def __init__(
        self, start_wp, max_distance, lane_width,
        choices=[], flip_x=False, flip_y=False
    ):
        """Constructor.
    
        Parameters
        ==========
        start_wp : carla.Waypoint
            Waypoint designating the start of the path.
        max_distance : float
            Maximum distance of path from start_wp we want to sample from.
        lane_width : float
            Size of road.
        choices : list of int
            Indices of turns at each junction along the path from start_wp onwards.
            If there are more junctions than indices contained in choices, then 
            choose default turn.
        """
        self.delta = lane_width / 2
        (
            self.waypoints, self.points, self.distances
        ) = carlautil.collect_points_along_path(
            start_wp, choices, max_distance,
            precision=self.__PRECISION,
            flip_x=flip_x, flip_y=flip_y
        )
        self.distance_to_point = pd.DataFrame(
            {
                "x": self.points[1:, 0],
                "y": self.points[1:, 1],
                "distances": self.distances[1:]},
            index=pd.IntervalIndex.from_breaks(self.distances)
        )
        self.junction_mask = self.__pad_junction_mask(
            util.map_to_ndarray(lambda wp: wp.is_junction, self.waypoints)
        )
        self.ids_df = self.__interval_index_ids(self.waypoints, self.distances)
        self.points_splits, self.split_mask = self.__split_line_by_mask(self.points, self.junction_mask)
        self.distances_splits, _ = self.__split_line_by_mask(self.distances, self.junction_mask)
        self.splines = []
        self.dsplines = []
        self.ddsplines = []
        for distances, points in zip(self.distances_splits, self.points_splits):
            spline = scipy.interpolate.CubicSpline(distances, points, axis=0)
            """Can make spline with another degree"""
            #self.spline = scipy.interpolate.make_interp_spline(
            #    self.distances, self.points, k=self.__K, axis=0
            #)
            """Can be LSQ B-spline, if waypoint points aren't aligned, but not necessary"""
            #t = np.linspace(self.distances[1], self.distances[-2], n_points // 2)
            #t = np.r_[(self.distances[0],)*(self.__K+1), t, (self.distances[-1],)*(self.__K+1)]
            #self.spline = scipy.interpolate.make_lsq_spline(
            #    self.distances, self.points, t=t, k=self.__K, axis=0
            #)
            dspline = spline.derivative(1)
            ddspline = spline.derivative(2)
            self.splines.append(spline)
            self.dsplines.append(dspline)
            self.ddsplines.append(ddspline)
        self.road_segs = self.cover_along_path_varyingsize()
        
    @property
    def path_length(self):
        return self.distances[-1]

    def get_point_from_start(self, distance):
        """Get goal point distance away from starting waypoint along path.
        
        Parameters
        ==========
        distance : float
            Distance away from starting waypoint.
        
        Returns
        =======
        ndarray
            (x, y) point distance away from starting waypoing along path.
        """
        for spline in self.splines:
            if spline.x[0] <= distance and distance <= spline.x[-1]:
                return spline(distance)
        return self.points[-1]
        
    def collect_segs_polytopes_and_goal(self, position, distance):
        """Collect the road boundary constraints and goal sufficient
        for the agent to go from a given position to a given distance
        further down the path.

        Parameters
        ==========
        position : ndarray
            Current position of agent.
        distance : float
            Distance from current agent position to move down the path.

        Returns
        =======
        util.AttrDict
            - polytopes : list of (tuple of ndarray)
                The segments covering sufficient parts of the path.
            - polytope_ids : list of int
                The IDs of the polytops corresponding to RoundBoundaryConstraint.road_segs
            - mask : ndarray of bool
                Mask of True whether a polytopes cover a junction, False otherwise.
            - goal : ndarray
                The final destination goal from current position.
        """
        beg_idx = np.argmin(np.linalg.norm(self.points - position, axis=1))
        beg_dist = self.distances[beg_idx]
        end_dist = min(self.distances[beg_idx] + distance, self.path_length)
        goal = self.distance_to_point.loc[end_dist][["x", "y"]].values
        beg_poly_id = max(self.road_segs.ids.loc[beg_dist].polytope_id - 1, 0)
        end_poly_id = min(
            self.road_segs.ids.loc[end_dist].polytope_id + 1,
            len(self.road_segs.polytopes) - 1
        )
        return util.AttrDict(
            polytopes=self.road_segs.polytopes[beg_poly_id:end_poly_id],
            polytope_ids=util.range_to_list(beg_poly_id, end_poly_id),
            mask=self.road_segs.mask[beg_poly_id:end_poly_id],
            goal=goal
        )

