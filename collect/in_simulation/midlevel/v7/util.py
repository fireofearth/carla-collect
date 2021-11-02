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
