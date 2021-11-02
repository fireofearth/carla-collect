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
from ....visualize.trajectron import render_map_crop

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
    actual_xy = actual_trajectory[:, :2]
    planned_frames, planned_trajectories = util.unzip([i for i in planned_trajectories.items()])
    goals = util.map_to_ndarray(lambda goal: [goal.x, goal.y], goals.values())
    
    N = len(planned_trajectories)
    fig, axes = plt.subplots(N // 2 + (N % 2), 2, figsize=(10, (10 / 4)*N))
    try:
        axes = axes.ravel()
    except AttributeError:
        axes = np.array([axes])
    
    for ax, planned_frame, planned_trajectory, goal \
            in zip(axes, planned_frames, planned_trajectories, goals):
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

    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))
    fig.clf()
