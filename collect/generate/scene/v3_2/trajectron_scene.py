"""Based on trajectron-plus-plus/experiments/nuScenes/process_data.py
"""

# Built-in packages
import sys
import os
import logging
import argparse

# PyPI packages
import dill
import numpy as np
import pandas as pd
import scipy
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from pyquaternion import Quaternion
from tqdm import tqdm

# Local packages
import utility as util

try:
    # trajectron-plus-plus/trajectron
    from environment import Environment, Scene, Node, GeometricMap, derivative_of
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

try:
    # trajectron-plus-plus/experiments/nuScenes
    from kalman_filter import NonlinearKinematicBicycle
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/experiments/nuScenes")

from ...label import SegmentationLabel
from ..scene import SceneBuilder
from ..scene import points_to_2d_histogram, round_to_int
from ..trajectron_util import (
        standardization, FREQUENCY, dt,
        data_columns_vehicle,
        data_columns_pedestrian,
        trajectory_curvature,
        plot_trajectron_scene)
from ...map.road import (
        remove_line_segments_by_condition,
        remove_polygons_by_condition)

curv_0_2 = 0
curv_0_1 = 0
total = 0
occlusion = 0

def print_and_reset_specs():
    global total
    global curv_0_2
    global curv_0_1
    global occlusion
    print(f"Total Nodes: {total}")
    print(f"Curvature > 0.1 Nodes: {curv_0_1}")
    print(f"Curvature > 0.2 Nodes: {curv_0_2}")
    print(f"(Between) occlusions encountered: {occlusion}")
    total = 0
    curv_0_1 = 0
    curv_0_2 = 0
    occlusion = 0

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == 'PEDESTRIAN':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()
            vx = node.data.velocity.x.copy()
            vy = node.data.velocity.y.copy()
            ax = node.data.acceleration.x.copy()
            ay = node.data.acceleration.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)
            vx, vy = rotate_pc(np.array([vx, vy]), alpha)
            ax, ay = rotate_pc(np.array([ax, ay]), alpha)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}

            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)
        elif node.type == 'VEHICLE':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()
            vx = node.data.velocity.x.copy()
            vy = node.data.velocity.y.copy()
            ax = node.data.acceleration.x.copy()
            ay = node.data.acceleration.y.copy()

            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            # sets heading in between [-pi, pi]
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)
            vx, vy = rotate_pc(np.array([vx, vy]), alpha)
            ax, ay = rotate_pc(np.array([ax, ay]), alpha)
            heading_x, heading_y = np.cos(heading), np.sin(heading)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): node.data.velocity.norm.copy(),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): node.data.acceleration.norm.copy(),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): getattr(node.data.heading, 'd°').copy()}

            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep,
                        non_aug_node=node)

        scene_aug.nodes.append(node)
    return scene_aug

def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    scene_aug.map = scene.map
    return scene_aug

def process_carla_scene(scene, data, max_timesteps, scene_config):
    for node_id in pd.unique(data['node_id']):
        # What is this?
        node_frequency_multiplier = 1
        node_df = data[data['node_id'] == node_id].copy()
        if node_df['x'].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df['frame_id']) == 1):
            """When there is occlusion, then take the longest
            subsequence of consecutive vehicle observations."""
            global occlusion
            occlusion += 1
            # plot_occlusion(scene, data, node_df, occl_count)
            s, sz = util.longest_consecutive_increasing_subsequence(node_df['frame_id'].values)
            logging.info(f"Found an occlusion by node {node_id} in scene {scene.name}.")
            logging.info(f"List of frame_id is {node_df['frame_id'].values}")
            occl_count = np.diff(node_df['frame_id'])
            logging.info(f"np.diff() on frame_id is {occl_count}; longest sequence is {sz}")
            if sz < 2:
                continue
            node_df = node_df[s].copy()
            logging.info(f"Sequence of frame_id is {node_df['frame_id'].values}")
            node_df['frame_id'] = np.arange(sz)

        if node_df.iloc[0]['type'] == scene_config.node_type.VEHICLE and not node_id == 'ego':
            x, y = node_df['x'].values, node_df['y'].values
            curvature, pl, _ = trajectory_curvature(np.stack((x, y), axis=-1))
            global total
            global curv_0_2
            global curv_0_1
            total += 1
            if pl > 1.0:
                if curvature > .2:
                    curv_0_2 += 1
                    node_frequency_multiplier = 3*int(np.floor(total/curv_0_2))
                elif curvature > .1:
                    curv_0_1 += 1
                    node_frequency_multiplier = 3*int(np.floor(total/curv_0_1))
        
        x, y = node_df['x'].values, node_df['y'].values
        v_x = derivative_of(x, scene.dt)
        v_y = derivative_of(y, scene.dt)
        a_x = derivative_of(v_x, scene.dt)
        a_y = derivative_of(v_y, scene.dt)
        if node_df.iloc[0]['type'] == scene_config.node_type.VEHICLE:
            v = np.stack((v_x, v_y,), axis=-1)
            a = np.stack((a_x, a_y,), axis=-1)
            v_norm = np.linalg.norm(v, axis=-1)
            a_norm = np.linalg.norm(a, axis=-1)
            heading = node_df['heading'].values
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi
            heading_x, heading_y = np.cos(heading), np.sin(heading)
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): v_x,
                         ('velocity', 'y'): v_y,
                         ('velocity', 'norm'): v_norm,
                         ('acceleration', 'x'): a_x,
                         ('acceleration', 'y'): a_y,
                         ('acceleration', 'norm'): a_norm,
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, scene.dt, radian=True)}
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        else:
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): v_x,
                         ('velocity', 'y'): v_y,
                         ('acceleration', 'x'): a_x,
                         ('acceleration', 'y'): a_y}
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)
        
        node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id,
                data=node_data, frequency_multiplier=node_frequency_multiplier)
        node.first_timestep = node_df['frame_id'].iloc[0]
        if node_df.iloc[0]['robot'] == True:
            node.is_robot = True
            scene.robot = node
        scene.nodes.append(node)
    return scene

class TrajectronPlusPlusSceneBuilder(SceneBuilder):
    # Should be large enough for map encoder to crop an input map around the vehicle
    MAP_PADDING = 120.
    DISTANCE_FROM_ROAD = 20.
    Z_MAP_UPPERBOUND = 6.
    Z_MAP_LOWERBOUND = 6.
    MAP_NAMES_TO_PREPROCESS = ['Town04', 'Town05']

    def __select_map_data(self, data, z_bound):
        if any(map(lambda s: data.map_name in s, self.MAP_NAMES_TO_PREPROCESS)):
            _z_bound = [
                    z_bound[0] - self.Z_MAP_LOWERBOUND,
                    z_bound[1] + self.Z_MAP_UPPERBOUND]
            road_polygons = remove_polygons_by_condition(
                    lambda a: (_z_bound[0] <= a[:,2]) & (a[:,2] <= _z_bound[1]), 
                    data.map_data.road_polygons)
            yellow_lines = remove_line_segments_by_condition(
                    lambda a: (_z_bound[0] <= a[:,2]) & (a[:,2] <= _z_bound[1]),
                    data.map_data.yellow_lines)
            white_lines = remove_line_segments_by_condition(
                    lambda a: (_z_bound[0] <= a[:,2]) & (a[:,2] <= _z_bound[1]),
                    data.map_data.white_lines)
            return road_polygons, yellow_lines, white_lines
        else:
            # skip time-consuming map data processing 
            return data.map_data.road_polygons, \
                    data.map_data.yellow_lines, \
                    data.map_data.white_lines

    def __process_carla_scene(self, data):
        # Get extent and sizing from trajectory data.
        traj_data = data.trajectory_data
        comp_key = np.logical_and(traj_data['node_id'] == 'ego', traj_data['frame_id'] == 0)
        ego_data = traj_data[comp_key]
        ego_z_bound = [ego_data['z'].min(), ego_data['z'].max()]
        s = ego_data.iloc[0]
        ego_initx, ego_inity = s['x'], s['y']
        max_timesteps = traj_data['frame_id'].max()
        x_min = round_to_int(traj_data['x'].min() - self.MAP_PADDING)
        x_max = round_to_int(traj_data['x'].max() + self.MAP_PADDING)
        y_min = round_to_int(traj_data['y'].min() - self.MAP_PADDING)
        y_max = round_to_int(traj_data['y'].max() + self.MAP_PADDING)
        x_size = x_max - x_min
        y_size = y_max - y_min
        
        # Filter road from LIDAR points.
        points = data.overhead_points
        z_mask = np.logical_and(
                points[:, 2] > ego_z_bound[0] - self.Z_LOWERBOUND,
                points[:, 2] < ego_z_bound[1] + self.Z_UPPERBOUND)
        points = data.overhead_points[z_mask]
        labels = data.overhead_labels[z_mask]
        road_label_mask = np.logical_or(labels == SegmentationLabel.Road.value,
                labels == SegmentationLabel.Vehicle.value)
        road_points = points[road_label_mask]
        
        # Adjust and filter occlusions from trajectory data.
        bitmap = points_to_2d_histogram(
                road_points, x_min, x_max, y_min, y_max,
                data.scene_config.pixels_per_m)
        bitmap[bitmap > 0] = 1
        traj_data['x'] = traj_data['x'] - x_min
        traj_data['y'] = traj_data['y'] - y_min
        transform = scipy.ndimage.distance_transform_cdt(-bitmap + 1)
        X = traj_data[['x', 'y']].values
        Xind = ( data.scene_config.pixels_per_m*X ).astype(int)
        vals = transform[ Xind.T[0], Xind.T[1] ]
        comp_key = np.logical_or(traj_data['node_id'] == 'ego', vals < self.DISTANCE_FROM_ROAD)
        traj_data = traj_data[comp_key]

        # Create bitmap from map data.
        pixels_per_m = data.scene_config.pixels_per_m
        road_polygons, yellow_lines, white_lines \
                = self.__select_map_data(data, ego_z_bound)
        dim = (int(pixels_per_m * y_size), int(pixels_per_m * x_size), 3)
        bitmap = np.zeros(dim)

        """NuScenes bitmap format
        scene.map[...].as_image() has shape (y, x, c)
        Channel 1: lane, road_segment, drivable_area
        Channel 2: road_divider
        Channel 3: lane_divider"""
        
        for polygon in road_polygons:
            rzpoly = ( pixels_per_m*(polygon[:,:2] - np.array([x_min, y_min])) ) \
                    .astype(int).reshape((-1,1,2))
            cv.fillPoly(bitmap, [rzpoly], (255,0,0))

        for line in yellow_lines:
            rzline = ( pixels_per_m*(line[:,:2] - np.array([x_min, y_min])) ) \
                    .astype(int).reshape((-1,1,2))
            cv.polylines(bitmap, [rzline], False, (0,255,0), thickness=2)

        for line in white_lines:
            rzline = ( pixels_per_m*(line[:,:2] - np.array([x_min, y_min])) ) \
                    .astype(int).reshape((-1,1,2))
            cv.polylines(bitmap, [rzline], False, (0,0,255), thickness=2)
        bitmap = bitmap.astype(np.uint8).transpose((2, 1, 0))

        # Create scene
        dt = data.fixed_delta_seconds * data.scene_config.record_interval
        scene = Scene(timesteps=max_timesteps + 1, dt=dt,
                name=data.scene_name, aug_func=augment)
        scene.ego_initx = ego_initx
        scene.ego_inity = ego_inity
        scene.x_min = x_min
        scene.y_min = y_min
        scene.x_max = x_max
        scene.y_max = y_max
        scene.x_size = x_size
        scene.y_size = y_size

        patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
        patch_angle = 0
        canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
        homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
        layer_names = ['drivable_area']

        scene.patch_box = patch_box
        scene.patch_angle = patch_angle
        scene.canvas_size = canvas_size
        scene.homography = homography
        scene.layer_names = layer_names

        type_map = dict()
        type_map['VEHICLE']       = GeometricMap(data=bitmap,
                homography=homography, description=', '.join(layer_names))
        type_map['VISUALIZATION'] = GeometricMap(data=bitmap,
                homography=homography, description=', '.join(layer_names))
        scene.map = type_map
        scene_config = data.scene_config
        return scene, traj_data, max_timesteps, scene_config

    def process_scene(self, data):
        scene, traj_data, max_timesteps, scene_config = self.__process_carla_scene(data)
        return process_carla_scene(scene, traj_data, max_timesteps, scene_config)
