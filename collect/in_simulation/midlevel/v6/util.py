# Built-in libraries
import os
import logging

# PyPI libraries
import numpy as np
import pandas as pd
import scipy.spatial
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm
import matplotlib.patches as patches
import control
import control.matlab
import docplex.mp
import docplex.mp.model

# Local libraries
import carla
import utility as util
import utility.npu
import carlautil
import carlautil.debug
from ....visualize.trajectron import render_scene
from ....trajectron import scene_to_df
from ..util import (get_vertices_from_center, get_ovehicle_color_set, plot_h_polyhedron)

# Profiling libraries
import functools
import cProfile, pstats, io

AGENT_COLORS = [
        'blue', 'darkviolet', 'dodgerblue', 'darkturquoise',
        'green', 'gold', 'orange', 'red', 'deeppink']
AGENT_COLORS = np.array(AGENT_COLORS) \
        .take([(i * 5) % len(AGENT_COLORS) for i in range(17)], 0)
NCOLORS = len(AGENT_COLORS)

###############################
# Methods for original approach
###############################

def plot_lcss_prediction_timestep(ax, scene, ovehicles,
        params, ctrl_result, X_star, t, ego_bbox, extent=None):
    ovehicle_colors = get_ovehicle_color_set()
    render_scene(ax, scene, global_coordinates=True)
    ax.plot(ctrl_result.goal[0], ctrl_result.goal[1],
            marker='*', markersize=8, color="green")

    # Plot ego vehicle
    ax.plot(X_star[:(t + 2), 0], X_star[:(t + 2), 1], 'k-o', markersize=2)

    # Get vertices of EV and plot its bounding box
    vertices = get_vertices_from_center(
            ctrl_result.X_star[t, :2],
            ctrl_result.X_star[t, 2],
            ego_bbox)
    bb = patches.Polygon(vertices.reshape((-1,2,)),
            closed=True, color='k', fc='none')
    ax.add_patch(bb)

    # Plot other vehicles
    for ov_idx, ovehicle in enumerate(ovehicles):
        color = ovehicle_colors[ov_idx][0]
        ax.plot(ovehicle.past[:,0], ovehicle.past[:,1],
                marker='o', markersize=2, color=color)
        for latent_idx in range(ovehicle.n_states):
            color = ovehicle_colors[ov_idx][latent_idx]
            
            # Plot overapproximation
            A = ctrl_result.A_union[t][latent_idx][ov_idx]
            b = ctrl_result.b_union[t][latent_idx][ov_idx]
            try:
                plot_h_polyhedron(ax, A, b, ec=color, alpha=1)
            except scipy.spatial.qhull.QhullError as e:
                print(f"Failed to plot polyhedron at timestep t={t}")

            # Plot vertices
            vertices = ctrl_result.vertices[t][latent_idx][ov_idx]
            X = vertices[:,0:2].T
            ax.scatter(X[0], X[1], color=color, s=2)
            X = vertices[:,2:4].T
            ax.scatter(X[0], X[1], color=color, s=2)
            X = vertices[:,4:6].T
            ax.scatter(X[0], X[1], color=color, s=2)
            X = vertices[:,6:8].T
            ax.scatter(X[0], X[1], color=color, s=2)

    if extent is not None:
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])
    ax.set_title(f"t = {t}")
    ax.set_aspect('equal')

def plot_lcss_prediction(pred_result, ovehicles,
        params, ctrl_result, T, ego_bbox, filename='lcss_control'):
    """
    Plot predictions and control trajectory for v1 and v2 controls.

    Parameters
    ==========
    predict_result : util.AttrDict
        Payload containing scene, timestep, nodes, predictions, z, latent_probs,
        past_dict, ground_truth_dict
    ovehicles : list of OVehicle
    params : util.AttrDict
    ctrl_result : util.AttrDict
    T : int
    ego_bbox : list of int
    filename : str
    """
    
    """Plots for paper"""
    fig, axes = plt.subplots(T // 2 + (T % 2), 2, figsize=(10, (10 / 4)*T))
    axes = axes.ravel()
    X_star = np.concatenate((params.initial_state.world[None], ctrl_result.X_star))
    x_min, y_min = np.min(X_star[:, :2], axis=0) - 20
    x_max, y_max = np.max(X_star[:, :2], axis=0) + 20
    extent = (x_min, x_max, y_min, y_max)
    for t, ax in zip(range(T), axes):
        plot_lcss_prediction_timestep(ax, pred_result.scene, ovehicles,
                params, ctrl_result, X_star, t, ego_bbox, extent=extent)
    
    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))
    fig.clf()

def plot_oa_simulation(scene, actual_trajectory, planned_trajectories,
        planned_controls, road_segs, ego_bbox, step_horizon,
        filename="oa_simulation", road_boundary_constraints=True):
    """Plot to compare between actual trajectory and planned trajectories"""

    scene_df = scene_to_df(scene)
    scene_df[['position_x', 'position_y']] += np.array([scene.x_min, scene.y_min])
    node_ids = scene_df['node_id'].unique()
    frames, actual_trajectory = util.unzip([i for i in actual_trajectory.items()])
    frames = np.array(frames)
    actual_trajectory = np.stack(actual_trajectory)
    actual_xy    = actual_trajectory[:, :2]
    actual_psi   = actual_trajectory[:, 2]
    actual_delta = actual_trajectory[:, 4]
    planned_frames, planned_trajectories = util.unzip([i for i in planned_trajectories.items()])
    
    N = len(planned_trajectories)
    fig, axes = plt.subplots(N, 3, figsize=(20, (10 / 2)*N))

    for ax in axes[:, 0]:
        """Render road overlay and plot actual trajectory"""
        render_scene(ax, scene, global_coordinates=True)
        if road_boundary_constraints:
            for A, b in road_segs.polytopes:
                util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.2)
    
    for idx, node_id in enumerate(node_ids):
        """Plot OV trajectory"""
        if node_id == "ego": continue
        node_df = scene_df[scene_df['node_id'] == node_id]
        X = node_df[['position_x', 'position_y']].values.T
        for ax in axes[:, 0]:
            ax.plot(X[0], X[1], ':.', color=AGENT_COLORS[idx % NCOLORS])
    
    for _axes, planned_frame, planned_trajectory in zip(axes, planned_frames, planned_trajectories):
        """Plot planned trajectories on separate plots"""
        ax = _axes[0]
        planned_xy = planned_trajectory[:, :2]
        ax.plot(planned_xy.T[0], planned_xy.T[1], "--b.", zorder=20)
        # plans are made in current frame, and carried out next frame
        idx = np.argwhere(frames == planned_frame)[0, 0] + 1
        ax.plot(actual_xy[:idx, 0], actual_xy[:idx, 1], '-ko')
        ax.plot(actual_xy[idx:(idx + step_horizon), 0], actual_xy[idx:(idx + step_horizon), 1], '-o', color="orange")
        ax.plot(actual_xy[(idx + step_horizon):, 0], actual_xy[(idx + step_horizon):, 1], '-ko')
        min_x, min_y = np.min(planned_xy, axis=0) - 20
        max_x, max_y = np.max(planned_xy, axis=0) + 20
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        """Plot psi, which is the vehicle longitudinal angle in global coordinates"""
        ax = _axes[1]
        planned_psi = planned_trajectory[:, 2]
        ax.plot(range(1, frames.size + 1), actual_psi, "-k.")
        ax.plot(range(idx, idx + planned_psi.size), planned_psi, "-b.")
        """Plot delta, which is the turning angle"""
        ax = _axes[2]
        planned_delta = planned_trajectory[:, 4]
        ax.plot(range(1, frames.size + 1), actual_delta, "-k.")
        ax.plot(range(idx, idx + planned_delta.size), planned_delta, "-b.")

    for ax in axes[:, 1:].ravel():
        ax.grid()

    axes[0, 0].set_title("Trajectories on map")
    axes[0, 1].set_title("$\Psi$ EV longitudinal angle, radians")
    axes[0, 2].set_title("$\delta$ EV turning angle, radians")

    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))
    fig.clf()
