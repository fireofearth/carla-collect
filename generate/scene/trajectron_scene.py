"""Based on trajectron-plus-plus/experiments/nuScenes/process_data.py
"""

import sys
import os
import logging
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import pandas as pd
import dill
import argparse
from tqdm import tqdm
from pyquaternion import Quaternion
from sklearn.model_selection import train_test_split

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

from ..label import SegmentationLabel
from .scene import SceneBuilder
from .scene import points_to_2d_histogram, round_to_int

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

curv_0_2 = 0
curv_0_1 = 0
total = 0

standardization = {
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}

def print_and_reset_specs():
    global total
    global curv_0_2
    global curv_0_1
    print(f"Total Nodes: {total}")
    print(f"Curvature > 0.1 Nodes: {curv_0_1}")
    print(f"Curvature > 0.2 Nodes: {curv_0_2}")
    total = 0
    curv_0_1 = 0
    curv_0_2 = 0

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

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

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

            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}

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

def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance

def process_trajectron_scene(scene, data, max_timesteps, scene_config):
    for node_id in pd.unique(data['node_id']):
        node_frequency_multiplier = 1
        node_df = data[data['node_id'] == node_id]

        if node_df['x'].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df['frame_id']) == 1):
            logging.info("Occlusion")
            logging.info(np.diff(node_df['frame_id']))
            continue  # TODO Make better

        node_values = node_df[['x', 'y']].values
        x = node_values[:, 0]
        y = node_values[:, 1]
        heading = node_df['heading'].values
        if node_df.iloc[0]['type'] == scene_config.node_type.VEHICLE and not node_id == 'ego':
            # Kalman filter Agent
            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            velocity = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1)

            filter_veh = NonlinearKinematicBicycle(dt=scene.dt, sMeasurement=1.0)
            P_matrix = None
            for i in range(len(x)):
                if i == 0:  # initalize KF
                    # initial P_matrix
                    P_matrix = np.identity(4)
                elif i < len(x):
                    # assign new est values
                    x[i] = x_vec_est_new[0][0]
                    y[i] = x_vec_est_new[1][0]
                    heading[i] = x_vec_est_new[2][0]
                    velocity[i] = x_vec_est_new[3][0]

                if i < len(x) - 1:  # no action on last data
                    # filtering
                    x_vec_est = np.array([[x[i]],
                                          [y[i]],
                                          [heading[i]],
                                          [velocity[i]]])
                    z_new = np.array([[x[i + 1]],
                                      [y[i + 1]],
                                      [heading[i + 1]],
                                      [velocity[i + 1]]])
                    x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                        x_vec_est=x_vec_est,
                        u_vec=np.array([[0.], [0.]]),
                        P_matrix=P_matrix,
                        z_new=z_new
                    )
                    P_matrix = P_matrix_new

            curvature, pl, _ = trajectory_curvature(np.stack((x, y), axis=-1))
            if pl < 1.0:  # vehicle is "not" moving
                x = x[0].repeat(max_timesteps + 1)
                y = y[0].repeat(max_timesteps + 1)
                heading = heading[0].repeat(max_timesteps + 1)
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

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        if node_df.iloc[0]['type'] == scene_config.node_type.VEHICLE:
            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
        else:
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

        node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data, frequency_multiplier=node_frequency_multiplier)
        node.first_timestep = node_df['frame_id'].iloc[0]
        if node_df.iloc[0]['robot'] == True:
            node.is_robot = True
            scene.robot = node

        scene.nodes.append(node)

    return scene


def plot_trajectron_scene(savedir, scene):
    fig, ax = plt.subplots(figsize=(15,15))
    # extent = (scene.x_min, scene.x_max, scene.y_min, scene.y_max)
    ax.imshow(scene.map['VEHICLE'].as_image(), origin='lower')
    spectral = cm.nipy_spectral(np.linspace(0, 1, len(scene.nodes)))
    for idx, node in enumerate(scene.nodes):
        # using values from scene.map['VEHICLE'].homography
        # to scale points
        xy = 3 * node.data.data[:, :2]
        ax.scatter(xy[:, 0], xy[:, 1], color=spectral[idx])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    fig.tight_layout()
    fn = f"{ scene.name.replace('/', '_') }.png"
    fp = os.path.join(savedir, fn)
    fig.savefig(fp)


# scipy.spatial.distance_matrix()


class TrajectronPlusPlusSceneBuilder(SceneBuilder):
    def process_scene(self, data):
        traj_data = data.trajectory_data
        # Trim points above/below certain Z levels.
        points = data.overhead_points
        z_mask = np.logical_and(
                points[:, 2] > self.Z_LOWERBOUND, points[:, 2] < self.Z_UPPERBOUND)
        points = data.overhead_points[z_mask]
        labels = data.overhead_labels[z_mask]
        # Select road LIDAR points.
        road_label_mask = labels == SegmentationLabel.Road.value
        road_points = points[road_label_mask]
        # Remove trajectory points that are far from road LIDAR points.
        X = traj_data[['x', 'y']].values
        D = scipy.spatial.distance_matrix(X, road_points[:, :2])
        traj_mask = np.min(D, axis=1) < 3.
        traj_data = traj_data[traj_mask]
        # Get extent and other data
        comp_key = np.logical_and(traj_data['node_id'] == 'ego', traj_data['frame_id'] == 0)
        s = traj_data[comp_key].iloc[0]
        ego_initx, ego_inity = s['x'], s['y']
        max_timesteps = traj_data['frame_id'].max()
        x_min = round_to_int(traj_data['x'].min() - 50)
        x_max = round_to_int(traj_data['x'].max() + 50)
        y_min = round_to_int(traj_data['y'].min() - 50)
        y_max = round_to_int(traj_data['y'].max() + 50)
        # Form bitmap
        bitmap = points_to_2d_histogram(
                road_points, x_min, x_max, y_min, y_max,
                data.scene_config.pixels_per_m)
        bitmap[bitmap > 0.] = 255.
        bitmap = np.stack((bitmap, np.zeros(bitmap.shape), np.zeros(bitmap.shape)), axis=0)
        bitmap = bitmap.astype(np.uint8)
        # adjust trajectory data
        traj_data['x'] = traj_data['x'] - x_min
        traj_data['y'] = traj_data['y'] - y_min
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

        type_map = dict()
        x_size = x_max - x_min
        y_size = y_max - y_min

        scene.map_name = data.map_name
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

        type_map['VEHICLE']       = GeometricMap(data=bitmap,
                homography=homography, description=', '.join(layer_names))
        type_map['VISUALIZATION'] = GeometricMap(data=bitmap,
                homography=homography, description=', '.join(layer_names))
        scene.map = type_map
        del bitmap
        return process_trajectron_scene(scene, traj_data, max_timesteps, data.scene_config)
