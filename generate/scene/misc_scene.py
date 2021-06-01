import os

import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

import utility as util

from .scene import SceneBuilder, points_to_2d_histogram, round_to_int
from ..label import SegmentationLabel


class BitmapSceneBuilder(SceneBuilder):
    def process_scene(self, data):
        # Remove trajectory points of hidden vehicles.
        comp_keys = pd.DataFrame.from_dict(data.vehicle_visibility, orient='index')
        comp_keys = comp_keys.stack().to_frame().reset_index().drop('level_1', axis=1)
        comp_keys.columns = ['frame_id', 'node_id']
        data.trajectory_data = pd.merge(data.trajectory_data, comp_keys,
                how='inner', on=['frame_id', 'node_id'])
        # Trim points above/below certain Z levels.
        points = data.overhead_points
        z_mask = np.logical_and(
                points[:, 2] > self.Z_LOWERBOUND, points[:, 2] < self.Z_UPPERBOUND)
        points = data.overhead_points[z_mask]
        labels = data.overhead_labels[z_mask]
        # Select road LIDAR points.
        road_label_mask = labels == SegmentationLabel.Road.value
        road_points = points[road_label_mask]
        # Get extent
        x_min = round_to_int(data.trajectory_data['x'].min() - 50)
        x_max = round_to_int(data.trajectory_data['x'].max() + 50)
        y_min = round_to_int(data.trajectory_data['y'].min() - 50)
        y_max = round_to_int(data.trajectory_data['y'].max() + 50)
        # Form bitmap
        bitmap = points_to_2d_histogram(
                road_points, x_min, x_max, y_min, y_max,
                data.scene_config.pixels_per_m)
        bitmap[bitmap > 0.] = 255.
        bitmap = np.stack((bitmap, np.zeros(bitmap.shape), np.zeros(bitmap.shape)), axis=-1)
        bitmap = bitmap.astype(np.uint8)
        # adjust trajectory data
        data.trajectory_data['x'] = data.trajectory_data['x'] - x_min
        data.trajectory_data['y'] = data.trajectory_data['y'] - y_min
        # Plot the data
        fig, ax = plt.subplots(figsize=(15,15))
        extent = (x_min, x_max, y_min, y_max)
        ax.imshow(bitmap.swapaxes(0, 1), extent=extent, origin='lower')
        trajdata = data.trajectory_data
        node_ids = trajdata[trajdata['type'] == 'VEHICLE']['node_id'].unique()
        #
        spectral = cm.nipy_spectral(np.linspace(0, 1, len(node_ids)))
        for idx, node_id in enumerate(node_ids):
            car_data = trajdata[trajdata['node_id'] == node_id]
            ax.scatter(car_data['x'] + x_min, car_data['y'] + y_min, color=spectral[idx])
        #
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fig.tight_layout()
        fn = f"{ data.scene_name.replace('/', '_') }.png"
        fp = os.path.join(data.save_directory, fn)
        fig.savefig(fp)
        return None


class PlotSceneBuilder(SceneBuilder):
    def process_scene(self, data):
        """
        TODO: it's not possible to extract some road lines from segmentation LIDAR
        """
        # Remove trajectory points of hidden vehicles.
        comp_keys = pd.DataFrame.from_dict(data.vehicle_visibility, orient='index')
        comp_keys = comp_keys.stack().to_frame().reset_index().drop('level_1', axis=1)
        comp_keys.columns = ['frame_id', 'node_id']
        data.trajectory_data = pd.merge(data.trajectory_data, comp_keys,
                how='inner', on=['frame_id', 'node_id'])
        # Trim points above/below certain Z levels.
        points = data.overhead_points
        z_mask = np.logical_and(
                points[:, 2] > self.Z_LOWERBOUND, points[:, 2] < self.Z_UPPERBOUND)
        points = data.overhead_points[z_mask]
        labels = data.overhead_labels[z_mask]
        # Select road LIDAR points.
        road_label_mask = labels == SegmentationLabel.Road.value
        road_points = points[road_label_mask]
        # Plot LIDAR and Trajectory points
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot()
        # Plot the road LIDAR points
        ax.scatter(road_points[:, 0], road_points[:, 1], s=2, c='black')
        # Plot the Trajectory points
        colors = ['green', 'yellow', 'orange', 'purple', 'pink']
        for idx, node_id in enumerate(data.trajectory_data['node_id'].unique()):
            trajectory_data = data.trajectory_data
            trajectory_data = trajectory_data[trajectory_data['node_id'] == node_id]
            color = colors[idx % 5] if node_id != 'ego' else 'red'
            ax.scatter(trajectory_data['x'], trajectory_data['y'], s=2, c=color)

            # Plot LIDAR points color coded by other vehicles
            # car_mask = data.overhead_ids == node_id
            # points = data.overhead_points[car_mask]
            # ax.scatter(points[:, 0], points[:, 1], s=2, c=color)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        fig.tight_layout()
        fn = f"{ data.scene_name.replace('/', '_') }.png"
        fp = os.path.join(data.save_directory, fn)
        fig.savefig(fp)
        return None

class DistanceTransformSceneBuilder(SceneBuilder):
    DISTANCE_FROM_ROAD = 10.

    def process_scene(self, data):
        # Remove trajectory points of hidden vehicles.
        # This method introduces occlusion to trajectories
        # comp_keys = pd.DataFrame.from_dict(data.vehicle_visibility, orient='index')
        # comp_keys = comp_keys.stack().to_frame().reset_index().drop('level_1', axis=1)
        # comp_keys.columns = ['frame_id', 'node_id']
        # data.trajectory_data = pd.merge(data.trajectory_data, comp_keys,
        #         how='inner', on=['frame_id', 'node_id'])

        # Trim points above/below certain Z levels.
        points = data.overhead_points
        z_mask = np.logical_and(
                points[:, 2] > self.Z_LOWERBOUND, points[:, 2] < self.Z_UPPERBOUND)
        points = data.overhead_points[z_mask]
        labels = data.overhead_labels[z_mask]
        # Select road LIDAR points.
        road_label_mask = labels == SegmentationLabel.Road.value
        road_points = points[road_label_mask]
        
        # Get extent and other data
        traj_data = data.trajectory_data
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
        bitmap[bitmap > 0.] = 1.
        nbitmap = -bitmap + 1
        dt = scipy.ndimage.distance_transform_cdt(nbitmap)

        dist = 15
        zbitmap = np.zeros(bitmap.shape)
        zbitmap[dt > self.DISTANCE_FROM_ROAD] = 1.
        zbitmap[bitmap == 1.] = 0.5

        # Plot the data
        fig, axes = plt.subplots(1, 2, figsize=(15,15))
        axes = axes.T
        extent = (x_min, x_max, y_min, y_max)
        axes[0].imshow(dt.T, extent=extent, origin='lower')
        axes[1].imshow(zbitmap.T, extent=extent, origin='lower')

        node_ids = traj_data[traj_data['type'] == 'VEHICLE']['node_id'].unique()
        #
        spectral = cm.nipy_spectral(np.linspace(0, 1, len(node_ids)))
        for idx, node_id in enumerate(node_ids):
            car_data = traj_data[traj_data['node_id'] == node_id]
            for ax in axes:
                ax.scatter(car_data['x'], car_data['y'], color=spectral[idx])
                ax.scatter(car_data['x'], car_data['y'], color=spectral[idx])
        #
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        fig.tight_layout()
        fn = f"{ data.scene_name.replace('/', '_') }.png"
        fp = os.path.join(data.save_directory, fn)
        fig.savefig(fp)
        return None

class DistanceTransformSelectSceneBuilder(SceneBuilder):
    DISTANCE_FROM_ROAD = 10.

    def process_scene(self, data):
        # Get extent and other data
        traj_data = data.trajectory_data
        comp_key = np.logical_and(traj_data['node_id'] == 'ego', traj_data['frame_id'] == 0)
        s = traj_data[comp_key].iloc[0]
        ego_initx, ego_inity = s['x'], s['y']
        max_timesteps = traj_data['frame_id'].max()
        x_min = round_to_int(traj_data['x'].min() - 50)
        x_max = round_to_int(traj_data['x'].max() + 50)
        y_min = round_to_int(traj_data['y'].min() - 50)
        y_max = round_to_int(traj_data['y'].max() + 50)
        x_size = x_max - x_min
        y_size = y_max - y_min
        traj_data['x'] = traj_data['x'] - x_min
        traj_data['y'] = traj_data['y'] - y_min

        # Trim points above/below certain Z levels.
        points = data.overhead_points
        z_mask = np.logical_and(
                points[:, 2] > self.Z_LOWERBOUND, points[:, 2] < self.Z_UPPERBOUND)
        points = data.overhead_points[z_mask]
        labels = data.overhead_labels[z_mask]
        # Select road LIDAR points.
        road_label_mask = labels == SegmentationLabel.Road.value
        road_points = points[road_label_mask]
        
        # Form bitmap
        bitmap = points_to_2d_histogram(
                road_points, x_min, x_max, y_min, y_max,
                data.scene_config.pixels_per_m)
        bitmap[bitmap > 0] = 1.

        # adjust trajectory data
        transform = scipy.ndimage.distance_transform_edt(-bitmap + 1)
        X = traj_data[['x', 'y']].values
        Xind = ( data.scene_config.pixels_per_m*X ).astype(int)
        vals = transform[ Xind.T[0], Xind.T[1] ]
        traj_data = traj_data[vals < self.DISTANCE_FROM_ROAD]

        bitmap[np.logical_and(transform < self.DISTANCE_FROM_ROAD, bitmap == 0)] = 0.5
        
        # Plot the data
        fig, axes = plt.subplots(1, 3, figsize=(15,15))
        axes = axes.T
        extent = (0, x_size, 0, y_size)
        axes[0].imshow(transform.T, extent=extent, origin='lower')
        axes[1].imshow(bitmap.T, extent=extent, origin='lower')

        node_ids = traj_data[traj_data['type'] == 'VEHICLE']['node_id'].unique()
        #
        spectral = cm.nipy_spectral(np.linspace(0, 1, len(node_ids)))
        for idx, node_id in enumerate(node_ids):
            car_data = traj_data[traj_data['node_id'] == node_id]
            for ax in axes[:2]:
                ax.scatter(car_data['x'], car_data['y'], color=spectral[idx])
                ax.scatter(car_data['x'], car_data['y'], color=spectral[idx])
            axes[2].scatter(Xind.T[0], Xind.T[1], color=spectral[idx])
        #
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        fig.tight_layout()
        fn = f"{ data.scene_name.replace('/', '_') }.png"
        fp = os.path.join(data.save_directory, fn)
        fig.savefig(fp)
        return None
