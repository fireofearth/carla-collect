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

# Import modules
from ....visualize.trajectron import render_map_crop
from ...dynamics.bicycle_v2 import compute_nonlinear_dynamical_states

# Local libraries
import carla
import utility as util
import utility.npu
import utility.plu
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
PADDING = 30

def plot_oa_simulation_timestep(
    map_data, actual_trajectory, frame_idx, planned_frame, planned_trajectory,
    planned_control, goal, road_segs, ego_bbox, step_horizon, steptime, n_frames,
    filename="oa_simulation",
    road_boundary_constraints=True,
    show_gt=False
):
    """Helper function of plot_oa_simulation()"""
    planned_xy = planned_trajectory[:, :2]
    planned_psi = planned_trajectory[:, 2]
    planned_v = planned_trajectory[:, 3]
    planned_a = planned_control[:, 0]
    planned_delta = planned_control[:, 1]
    actual_xy = actual_trajectory[:, :2]
    actual_psi  = actual_trajectory[:, 2]
    actual_v  = actual_trajectory[:, 3]
    min_x, min_y = np.min(planned_xy, axis=0)
    max_x, max_y = np.max(planned_xy, axis=0)
    x_mid, y_mid = (max_x + min_x) / 2, (max_y + min_y) / 2
    extent = (x_mid - PADDING, x_mid + PADDING, y_mid - PADDING, y_mid + PADDING)
    gt_planned_xy = None
    gt_planned_psi = None
    gt_planned_v = None
    if show_gt:
        gt_planned_trajectory = compute_nonlinear_dynamical_states(
            planned_trajectory[0], planned_trajectory.shape[0] - 1, steptime,
            planned_control, l_r=0.5*ego_bbox[0], L=ego_bbox[0]
        )
        gt_planned_xy = gt_planned_trajectory[:, :2]
        gt_planned_psi = gt_planned_trajectory[:, 2]
        gt_planned_v = gt_planned_trajectory[:, 3]

    # Generate plots for map, state and inputs
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    ax = axes[0]

    """Map overlay"""
    render_map_crop(ax, map_data, extent)
    if road_boundary_constraints:
        for A, b in road_segs.polytopes:
            util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.2)
    ax.plot(goal.x, goal.y, marker='*', markersize=8, color="yellow")
    
    "EV bounding box at current position"
    position = actual_trajectory[frame_idx, :2]
    heading = actual_trajectory[frame_idx, 2]
    vertices = util.npu.vertices_from_bbox(position, heading, ego_bbox)
    bb = patches.Polygon(vertices.reshape((-1,2,)),
            closed=True, color="k", fc=util.plu.modify_alpha("black", 0.2))
    ax.add_patch(bb)

    """Plot EV planned trajectory. This is computed from CPLEX using linearized
    discrete-time vehicle dynamics."""
    ax.plot(*planned_xy.T, "--bo", zorder=20, markersize=2)

    """Plot EV planned trajectory. This is the ground-truth non-linear vehicle dyamics
    computed by solving the model with ZOH and the controls computed from CPLEX."""
    if show_gt:
        ax.plot(*gt_planned_xy.T, "--go", zorder=20, markersize=2)

    """Plot EV actual trajectory.
    Plans are made in current frame, and carried out next frame."""
    idx = frame_idx + 1
    ax.plot(*actual_xy[:idx].T, '-ko')
    ax.plot(*actual_xy[idx:(idx + step_horizon)].T, '-o', color="orange")
    ax.plot(*actual_xy[(idx + step_horizon):].T, '-ko')

    """Plot bounding boxes along with trajectories on xy-plane."""
    ax = axes[1]
    ax.plot(*planned_xy.T, "--bo", zorder=20, markersize=2)
    ax.plot(*actual_xy.T, '-ko')
    vertices = util.npu.vertices_of_bboxes(actual_xy, actual_psi, ego_bbox)
    for v in vertices:
        bb = patches.Polygon(v,
                closed=True, color="k", fc=util.plu.modify_alpha("black", 0.2))
        ax.add_patch(bb)
    vertices = util.npu.vertices_of_bboxes(planned_xy, planned_psi, ego_bbox)
    for v in vertices:
        bb = patches.Polygon(v,
                closed=True, color="b", fc=util.plu.modify_alpha("blue", 0.2))
        ax.add_patch(bb)

    "Configure overhead plot."
    for ax in axes[:2]:
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"frame {frame_idx + 1}")
        ax.set_aspect('equal')

    """Plot v, which is the vehicle speed."""
    ax = axes[2]
    ax.plot(range(1, n_frames + 1), actual_v, "-k.", label="actual")
    ax.plot(range(idx, idx + planned_v.size), planned_v, "-b.", label="linear plan")
    if show_gt:
        ax.plot(
            range(idx, idx + gt_planned_v.size),
            gt_planned_v, "-g.", label="non-linear plan"
        )
    ax.set_title("$v$ speed of c.g., m/s")
    ax.set_ylabel("m/s")

    """Plot psi, which is the vehicle longitudinal angle in global coordinates."""
    ax = axes[3]
    ax.plot(range(1, n_frames + 1), actual_psi, "-k.", label="actual")
    ax.plot(range(idx, idx + planned_psi.size), planned_psi, "-b.", label="linear plan")
    if show_gt:
        ax.plot(
            range(idx, idx + gt_planned_psi.size),
            gt_planned_psi, "-g.", label="non-linear plan"
        )
    ax.set_title("$\psi$ longitudinal angle, radians")
    ax.set_ylabel("rad")

    """Plot delta, which is the turning angle.
    Not applicable."""
    # axes[1].set_visible(False)

    """Plot a, which is acceleration control input"""
    ax = axes[4]
    ax.plot(
        range(idx, idx + planned_a.size), planned_a,
        "-.", color="orange", label="control plan"
    )
    ax.set_title("$a$ acceleration input, $m/s^2$")
    # ax.set_ylabel("$m/s^2$")

    """Plot delta, which is steering control input"""
    ax = axes[5]
    ax.plot(
        range(idx, idx + planned_delta.size), planned_delta,
        "-.", color="orange", label="control plan"
    )
    ax.set_title("$\delta$ steering input, radians")
    # ax.set_ylabel("rad")

    for ax in axes[2:]:
        ax.set_xlabel("timestep")
        ax.grid()
        ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}_idx{frame_idx}.png"))
    fig.clf()


def plot_oa_simulation(
    map_data, actual_trajectory, planned_trajectories, planned_controls, goals,
    lowlevel, road_segs, ego_bbox, step_horizon, steptime, filename="oa_simulation",
    road_boundary_constraints=True
):
    """Plot to compare between actual trajectory and planned trajectories.
    Produces one plot per MPC step. Produces plots for velocity, steering.
    
    Parameters
    ==========
    map_data : util.AttrDict
        Container of vertices for road segments and lines produced by MapQuerier.
    actual_trajectory : collections.OrderedDict of (int, ndarray)
        Indexed by frame ID. Array of EV global (x, y) position, heading and speed.
    planned_trajectories : collections.OrderedDict of (int, ndarray)
        Indexed by frame ID. The EV's planned state over timesteps T+1 incl. origin
        with global (x, y) position, heading and speed as ndarray of shape (4, T + 1).
    planned_controls : collections.OrderedDict of (int, ndarray)
        Indexed by frame ID. The EV's planned controls over timesteps T with
        acceleration, steering as ndarray of shape (2, T).
    goals : collections.OrderedDict of (int, util.AttrDict)
        Indexed by frame ID. EV's goal.
    road_segs : util.AttrDict
        Container of road segment properties.
    ego_bbox : ndarray
        EV's longitudinal and lateral dimensions.
    step_horizon : int
        Number of predictions steps to execute at each iteration of MPC.
    steptime : float
        Time in seconds taken to complete one step of MPC.
    filename : str
        Partial file name to save plots.
    road_boundary_constraints : bool
        Whether to visualize boundary constrains in plots.
    """
    frames, actual_trajectory = util.unzip([i for i in actual_trajectory.items()])
    frames = np.array(frames)
    actual_trajectory = np.stack(actual_trajectory)
    planned_frames, planned_controls = util.unzip([i for i in planned_controls.items()])
    _, planned_trajectories = util.unzip([i for i in planned_trajectories.items()])
    _, goals = util.unzip([i for i in goals.items()])
    for planned_frame, planned_trajectory, planned_control, goal \
            in zip(planned_frames, planned_trajectories, planned_controls, goals):
        frame_idx = np.argwhere(frames == planned_frame)[0, 0]
        plot_oa_simulation_timestep(
            map_data, actual_trajectory, frame_idx, planned_frame, planned_trajectory,
            planned_control, goal, road_segs, ego_bbox, step_horizon, steptime,
            frames.size, filename=filename,
            road_boundary_constraints=road_boundary_constraints
        )
    
    # plot PID controller / actuation
    _, lowlevel = util.unzip([i for i in lowlevel.items()])
    m_speed = util.map_to_ndarray(lambda x: x.measurement.speed, lowlevel)
    r_speed = util.map_to_ndarray(lambda x: x.reference.speed, lowlevel)
    m_angle = util.map_to_ndarray(lambda x: x.measurement.angle, lowlevel)
    r_angle = util.map_to_ndarray(lambda x: x.reference.angle, lowlevel)
    m_angle = util.npu.warp_radians_neg_pi_to_pi(m_angle)
    r_angle = util.npu.warp_radians_neg_pi_to_pi(r_angle)
    c_throttle = util.map_to_ndarray(lambda x: x.control.throttle, lowlevel)
    c_brake = util.map_to_ndarray(lambda x: x.control.brake, lowlevel)
    c_steer = util.map_to_ndarray(lambda x: x.control.steer, lowlevel)
    pe_speed = util.map_to_ndarray(lambda x: x.error.speed.pe, lowlevel)
    ie_speed = util.map_to_ndarray(lambda x: x.error.speed.ie, lowlevel)
    de_speed = util.map_to_ndarray(lambda x: x.error.speed.de, lowlevel)
    pe_angle = util.map_to_ndarray(lambda x: x.error.angle.pe, lowlevel)
    ie_angle = util.map_to_ndarray(lambda x: x.error.angle.ie, lowlevel)
    de_angle = util.map_to_ndarray(lambda x: x.error.angle.de, lowlevel)

    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    axes = axes.T.ravel()

    steptimes = np.arange(len(lowlevel)) * 0.05
    ax = axes[0]
    ax.plot(steptimes, m_speed, "-g.", label="measured speed")
    ax.plot(steptimes, r_speed, "-", marker='.', color="orange", label="reference speed")
    ax.set_title("speed")
    ax.set_ylabel("m/s")

    ax = axes[1]
    ax.plot(steptimes, pe_speed, "-r.", label="prop. error")
    ax.plot(steptimes, ie_speed, "-b.", label="integral error")
    ax.plot(steptimes, de_speed, "-g.", label="derivative error")
    ax.set_title("speed error")
    ax.set_ylim([-10, 10])

    ax = axes[2]
    ax.plot(steptimes, c_throttle, "-b.", label="applied throttle")
    ax.plot(steptimes, c_brake, "-r.", label="applied throttle")
    ax.set_title("longitudinal control")

    ax = axes[3]
    ax.plot(steptimes, m_angle, "-g.", label="measured angle")
    ax.plot(steptimes, r_angle, "-", marker='.', color="orange", label="reference angle")
    ax.set_title("angle")
    ax.set_ylabel("rad")

    ax = axes[4]
    ax.plot(steptimes, pe_angle, "-r.", label="prop. error")
    ax.plot(steptimes, ie_angle, "-b.", label="integral error")
    ax.plot(steptimes, de_angle, "-g.", label="derivative error")
    ax.set_ylim([-10, 10])
    ax.set_title("angle error")

    ax = axes[5]
    ax.plot(steptimes, c_steer, "-b.", label="applied steer")
    ax.set_title("lateral control")

    for ax in axes:
        ax.grid()
        ax.legend()
        ax.set_xlabel("seconds s")

    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}_pid.png"))
    fig.clf()

