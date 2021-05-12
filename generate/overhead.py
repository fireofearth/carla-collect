import numpy as np
import carla

import utility as util
import carlautil

def splat_points(points, splat_params, nd=2):
    meters_max = splat_params.meters_max
    pixels_per_meter = splat_params.pixels_per_meter
    hist_max_per_pixel = splat_params.hist_max_per_pixel
    # meters_max = splat_params['meters_max']
    # pixels_per_meter = splat_params['pixels_per_meter']
    # hist_max_per_pixel = splat_params['hist_max_per_pixel']
    
    # Allocate 2d histogram bins. Todo tmp?
    ymeters_max = meters_max
    xbins = np.linspace(-meters_max, meters_max+1, meters_max * 2 * pixels_per_meter + 1)
    ybins = np.linspace(-meters_max, ymeters_max+1, ymeters_max * 2 * pixels_per_meter + 1)
    hist = np.histogramdd(points[..., :nd], bins=(xbins, ybins))[0]
    # Compute histogram of x and y coordinates of points
    # hist = np.histogram2d(x=points[:,0], y=points[:,1], bins=(bins, ybins))[0]

    # Clip histogram 
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel

    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel

    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground
    return overhead_splat


def get_normalized_sensor_data(lidar_measurement):
    """Obtain LIDAR point cloud with rotation oriented to
    world frame of reference, and centered at (0, 0, 0). 

    First convert to world orientation which keeping the origin
    using lidar_measurement.transform, mentioned in:
    https://github.com/carla-simulator/carla/issues/2817

    Rotate points 90 degrees CCW around origin (why?).

    Adjust points 2.5 meters in the z direction to reflect
    how the sensor is placed with resp. to the ego vehicle.

    Lastly reflect the y-axis, mentioned in:
    https://github.com/carla-simulator/carla/issues/2699
    
    Parameters
    ----------
    lidar_measurement : carla.LidarMeasurement or carla.SemanticLidarMeasurement

    Returns
    -------
    np.array
        Has shape (number of points, 3)
        corresponding to (x, y, z) coordinates.
    np.array
        Labels of points. Has shape (number of points,)
    
    TODO: still don't really understand what's going on with
    the transformation matrix produced by transformation.
    Check out open3d_lidar.py for fix
    """
    raw_data = lidar_measurement.raw_data
    if isinstance(lidar_measurement, carla.LidarMeasurement):
        """Array of 32-bits floats (XYZI of each point).
        """
        points = np.frombuffer(raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        labels = np.zeros(points.shape[0], dtype=np.uint32)
    elif isinstance(lidar_measurement, carla.SemanticLidarMeasurement):
        """Array containing the point cloud with instance and semantic information.
        For each point, four 32-bits floats are stored.
        XYZ coordinates.
        cosine of the incident angle.
        Unsigned int containing the index of the object hit.
        Unsigned int containing the semantic tag of the object it.
        """
        data = np.frombuffer(raw_data, dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
        points = np.array([data['x'], data['y'], data['z'], data['CosAngle']]).T
        labels = data['ObjTag']
    else:
        raise ValueError("Unknown lidar measurement.")
    
    transform_to_world = carla.Transform(
            carla.Location(),
            lidar_measurement.transform.rotation)
    transform_ccw = carla.Transform(
            carla.Location(),
            carla.Rotation(yaw=-90.))
    matrix = np.array(transform_to_world.get_matrix())
    points = np.dot(matrix, points.T).T
    matrix = np.array(transform_ccw.get_matrix())
    points = np.dot(matrix, points.T).T
    points = points[:, :3] \
        + carlautil.location_to_ndarray(carla.Location(z=2.5))
    points[:, 1] *= -1.0
    return points, labels


def old_get_occupancy_grid(points, lidar_params, player_bbox):
    """
    based on carla_preprocess.get_occupancy_grid

    Parameters
    ----------
    lidar_sensor : carla.Sensor - a sensor from blueprint sensor.lidar.ray_cast
    lidar_measurement : carla.LidarMeasurement
    player_transform : carla.Transform
    lidar_params : LidarParams

    """
    # tag points above 0.1 meters as non-road obstacles.
    z_threshold = 0.1
    # tage points above vehicle as above obstacles.
    z_threshold_second_above = player_bbox.extent.z * 2
    above_mask = points[:, 2] > z_threshold
    second_above_mask = points[:, 2] > z_threshold_second_above

    meters_max = lidar_params.meters_max
    pixels_per_meter = lidar_params.pixels_per_meter
    val_obstacle = lidar_params.val_obstacle
    def get_occupancy_from_masked_lidar(mask):
        masked_lidar = points[mask]
        xbins = np.linspace(-meters_max, meters_max, meters_max * 2 * pixels_per_meter + 1)
        ybins = xbins
        grid = np.histogramdd(masked_lidar[..., :2], bins=(xbins, ybins))[0]
        grid[grid > 0.] = val_obstacle
        return grid

    above = get_occupancy_from_masked_lidar(above_mask)
    below = get_occupancy_from_masked_lidar(np.logical_not(above_mask))
    second_above = get_occupancy_from_masked_lidar(second_above_mask)
    feats = (above, below, second_above,)
    return np.stack(feats, axis=-1)


def get_occupancy_grid(points, lidar_params, player_bbox, labels):
    """Create occupancy grid of below (ground) and above (obstacles) points.
    Uses segmentation LIDAR labels to tag road points into ground points.

    Labels:

    6 RoadLine - The markings on the road
    7 Road - Part of ground on which cars usually drive (E.g. lanes in any directions, and streets.

    Parameters
    ----------
    points : np.array
        Has shape (number of points, 3)
        corresponding to (x, y, z) coordinates.
    lidar_params : LidarParams
    player_bbox : carla.BoundingBox
    labels : np.array
        Has shape (number of points,)
    """
    # tag points above 0.06 meters as non-road obstacles.
    z_threshold = 0.06
    # tag points above vehicle as above obstacles.
    z_threshold_second_above = player_bbox.extent.z * 2 + 0.5
    road_mask = np.logical_or(labels == 6, labels == 7)
    below_mask = np.logical_or(road_mask, points[:, 2] <= z_threshold)

    all_above_mask = np.logical_not(below_mask)
    above_mask = np.logical_and(
            all_above_mask, points[:, 2] <= z_threshold_second_above)
    second_above_mask = np.logical_and(
            all_above_mask, points[:, 2] > z_threshold_second_above)

    meters_max = lidar_params.meters_max
    pixels_per_meter = lidar_params.pixels_per_meter
    val_obstacle = lidar_params.val_obstacle
    def get_occupancy_from_masked_lidar(mask):
        masked_lidar = points[mask]
        xbins = np.linspace(-meters_max, meters_max, meters_max * 2 * pixels_per_meter + 1)
        ybins = xbins
        grid = np.histogramdd(masked_lidar[..., :2], bins=(xbins, ybins))[0]
        grid[grid > 0.] = val_obstacle
        return grid

    below = get_occupancy_from_masked_lidar(below_mask)
    above = get_occupancy_from_masked_lidar(above_mask)
    second_above = get_occupancy_from_masked_lidar(second_above_mask)
    feats = (above, below, second_above,)
    return np.stack(feats, axis=-1)


def build_BEV(lidar_points, lidar_params, player_bbox,
        lidar_point_labels=None):
    """
    based on carla_preprocess.build_BEV

    Parameters
    ----------
    lidar_points : np.array
        Has shape (number of points, 3)
        corresponding to (x, y, z) coordinates.
    lidar_params : LidarParams
    player_bbox : carla.BoundingBox
    """
    if lidar_point_labels is None:
        lidar_point_labels = np.zeros(points.shape[0], dtype=np.unint32)
    overhead_lidar = splat_points(lidar_points, lidar_params)
    overhead_lidar_features = overhead_lidar[..., None]
    ogrid = get_occupancy_grid(lidar_points, lidar_params,
            player_bbox, lidar_point_labels)
    overhead_features = np.concatenate((overhead_lidar_features, ogrid), axis=-1)
    return overhead_features
