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

from ....visualize.trajectron import render_map_crop

# Local libraries
import carla
import utility as util
import utility.npu
import carlautil
import carlautil.debug

# Profiling libraries
import functools
import cProfile, pstats, io

AGENT_COLORS = [
        'blue', 'darkviolet', 'dodgerblue', 'darkturquoise',
        'green', 'gold', 'orange', 'red', 'deeppink']
AGENT_COLORS = np.array(AGENT_COLORS) \
        .take([(i * 5) % len(AGENT_COLORS) for i in range(17)], 0)
NCOLORS = len(AGENT_COLORS)

def plot_oa_simulation(map_data, actual_trajectory, planned_trajectories,
        planned_controls, goals, road_segs, ego_bbox, step_horizon,
        filename="oa_simulation", road_boundary_constraints=True):
    """Plot to compare between actual trajectory and planned trajectories"""

    frames, actual_trajectory = util.unzip([i for i in actual_trajectory.items()])
    frames = np.array(frames)
    actual_trajectory = np.stack(actual_trajectory)
    actual_xy    = actual_trajectory[:, :2]
    actual_psi   = actual_trajectory[:, 2]
    actual_v   = actual_trajectory[:, 3]
    actual_delta = actual_trajectory[:, 4]
    planned_frames, planned_trajectories = util.unzip([i for i in planned_trajectories.items()])
    goals = util.map_to_ndarray(util.identity, goals.values())
    
    N = len(planned_trajectories)
    fig, axes = plt.subplots(N, 4, figsize=(20, (10 / 2)*N))
    if N == 1:
        axes = axes[None]
    
    for _axes, planned_frame, planned_trajectory, goal \
            in zip(axes, planned_frames, planned_trajectories, goals):
        ax = _axes[0]
        planned_xy = np.concatenate((
            planned_trajectory[:, :2], goal[None],))
        min_x, min_y = np.min(planned_xy, axis=0) - 20
        max_x, max_y = np.max(planned_xy, axis=0) + 20
        extent = (min_x, max_x, min_y, max_y)
        planned_xy = planned_trajectory[:, :2]

        """Map overlay"""
        render_map_crop(ax, map_data, extent)
        if road_boundary_constraints:
            for A, b in road_segs.polytopes:
                util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.2)
        
        ax.plot(goal[0], goal[1], marker='*', markersize=8, color="green")

        """Plot planned trajectories on separate plots"""
        ax.plot(planned_xy.T[0], planned_xy.T[1], "--b.", zorder=20)
        # plans are made in current frame, and carried out next frame
        idx = np.argwhere(frames == planned_frame)[0, 0] + 1
        ax.plot(actual_xy[:idx, 0], actual_xy[:idx, 1], '-ko')
        ax.plot(actual_xy[idx:(idx + step_horizon), 0], actual_xy[idx:(idx + step_horizon), 1], '-o', color="orange")
        ax.plot(actual_xy[(idx + step_horizon):, 0], actual_xy[(idx + step_horizon):, 1], '-ko')

        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        """Plot, v, which is speed"""
        ax = _axes[1]
        planned_v = planned_trajectory[:, 3]
        ax.plot(range(1, frames.size + 1), actual_v, "-k.")
        ax.plot(range(idx, idx + planned_v.size), planned_v, "-b.")

        """Plot psi, which is the vehicle longitudinal angle in global coordinates"""
        ax = _axes[2]
        planned_psi = planned_trajectory[:, 2]
        ax.plot(range(1, frames.size + 1), actual_psi, "-k.")
        ax.plot(range(idx, idx + planned_psi.size), planned_psi, "-b.")

        """Plot delta, which is the turning angle"""
        ax = _axes[3]
        planned_delta = planned_trajectory[:, 4]
        ax.plot(range(1, frames.size + 1), actual_delta, "-k.")
        ax.plot(range(idx, idx + planned_delta.size), planned_delta, "-b.")

    for ax in axes[:, 1:].ravel():
        ax.grid()

    axes[0, 0].set_title("Trajectories on map")
    axes[0, 1].set_title("$v$ EV speed, m/s")
    axes[0, 2].set_title("$\Psi$ EV longitudinal angle, radians")
    axes[0, 3].set_title("$\delta$ EV turn angle, radians")

    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))
    fig.clf()
