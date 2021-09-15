"""
"""

import collections

import numpy as np

import carla
import utility as util
import carlautil

PRECISION = 1.0

def to_point(wp):
    """Convert waypoint to a 2D point.

    Parameters
    ==========
    wp : carla.Waypoint

    Returns
    =======
    list of float
    """
    x, y, _ = carlautil.to_location_ndarray(wp)
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
