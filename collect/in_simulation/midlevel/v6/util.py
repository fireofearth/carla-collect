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
from ....visualize.trajectron import render_scene
# from ....trajectron import scene_to_df
from ..dynamics import compute_nonlinear_dynamical_states
from ..util import get_ovehicle_color_set

import carla
import utility as util
import utility.npu
import utility.plu
import carlautil
import carlautil.debug

AGENT_COLORS = [
        'blue', 'darkviolet', 'dodgerblue', 'darkturquoise',
        'green', 'gold', 'orange', 'red', 'deeppink']
AGENT_COLORS = np.array(AGENT_COLORS) \
        .take([(i * 5) % len(AGENT_COLORS) for i in range(17)], 0)
NCOLORS = len(AGENT_COLORS)

def node_to_df(node):
    columns = ['_'.join(t) for t in node.data.header]
    return pd.DataFrame(node.data.data, columns=columns)

def scene_to_df(scene):
    dfs = [node_to_df(node) for node in scene.nodes if repr(node.type) == 'VEHICLE']
    tmp_dfs = []
    for node, df in zip(scene.nodes, dfs):
        df.insert(0, 'node_id', str(node.id))
        df.insert(0, 'frame_id', np.arange(len(df)) + node.first_timestep)
        tmp_dfs.append(df)
    return pd.concat(tmp_dfs)

###############################
# Methods for original approach
###############################

def plot_lcss_prediction_timestep(ax, pred_result, ovehicles,
        params, ctrl_result, X_star, t, ego_bbox, extent=None):
    ovehicle_colors = get_ovehicle_color_set()
    render_scene(ax, pred_result.scene, global_coordinates=True)
    ax.plot(ctrl_result.goal[0], ctrl_result.goal[1],
            marker='*', markersize=8, color="yellow")

    # Plot ego vehicle past trajectory
    past = None
    minpos = np.array([pred_result.scene.x_min, pred_result.scene.y_min])
    for node in pred_result.nodes:
        if node.id == 'ego':
            past = pred_result.past_dict[pred_result.timestep][node] + minpos
            break
    ax.plot(past[:, 0], past[:, 1], '-ko', markersize=2)

    # Box of current ego vehicle position
    vertices = util.vertices_from_bbox(
            params.initial_state.world[:2],
            params.initial_state.world[2],
            ego_bbox)
    bb = patches.Polygon(vertices.reshape((-1,2,)),
            closed=True, color='k', fc=util.plu.modify_alpha("black", 0.2), ls='-')
    ax.add_patch(bb)

    # Plot ego vehicle planned trajectory
    ax.plot(X_star[:(t + 2), 0], X_star[:(t + 2), 1], '--ko', markersize=2)
    
    # Get vertices of EV and plot its bounding box
    if t < 2:
        vertices = util.vertices_from_bbox(
                ctrl_result.X_star[t, :2], ctrl_result.X_star[t, 2], ego_bbox)
    else:
        # Make EV bbox look better on far horizons by fixing the heading angle
        heading = np.arctan2(ctrl_result.X_star[t, 1] - ctrl_result.X_star[t - 1, 1],
                ctrl_result.X_star[t, 0] - ctrl_result.X_star[t - 1, 0])
        vertices = util.vertices_from_bbox(
                ctrl_result.X_star[t, :2], heading, ego_bbox)

    bb = patches.Polygon(vertices.reshape((-1,2,)),
            closed=True, color='k', fc='none', ls='-')
    ax.add_patch(bb)

    # Plot other vehicles
    for ov_idx, ovehicle in enumerate(ovehicles):
        color = ovehicle_colors[ov_idx][0]
        ax.plot(ovehicle.past[:,0], ovehicle.past[:,1],
                marker='o', markersize=2, color=color)
        heading = np.arctan2(ovehicle.past[-1,1] - ovehicle.past[-2,1],
                ovehicle.past[-1,0] - ovehicle.past[-2,0])
        vertices = util.vertices_from_bbox(
                ovehicle.past[-1], heading, np.array([ovehicle.bbox]))
        bb = patches.Polygon(vertices.reshape((-1,2,)),
                closed=True, color=color, fc=util.plu.modify_alpha(color, 0.2), ls='-')
        ax.add_patch(bb)
        for latent_idx in range(ovehicle.n_states):
            color = ovehicle_colors[ov_idx][latent_idx]
            
            # Plot overapproximation
            A = ctrl_result.A_union[t][latent_idx][ov_idx]
            b = ctrl_result.b_union[t][latent_idx][ov_idx]
            try:
                util.npu.plot_h_polyhedron(ax, A, b, ec=color, ls='-', alpha=1)
            except scipy.spatial.qhull.QhullError:
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"t = {t + 1}")
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
    # make the plotting grid tall
    # fig, axes = plt.subplots(T // 2 + (T % 2), 2, figsize=(10, (10 / 4)*T))
    # make the plotting grid wide
    fig, axes = plt.subplots(2, T // 2 + (T % 2), figsize=((10 / 4)*T, 10))
    axes = axes.ravel()
    X_star = np.concatenate((params.initial_state.world[None], ctrl_result.X_star))
    x_min, y_min = np.min(X_star[:, :2], axis=0) - 20
    x_max, y_max = np.max(X_star[:, :2], axis=0) + 20
    # make extent fit the planned trajectory
    # extent = (x_min, x_max, y_min, y_max)
    # make extent the same size across all timesteps, with planned trajectory centered
    x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
    extent = (x_mid - 30, x_mid + 30, y_mid - 30, y_mid + 30)
    for t, ax in zip(range(T), axes):
        plot_lcss_prediction_timestep(ax, pred_result, ovehicles,
                params, ctrl_result, X_star, t, ego_bbox, extent=extent)
    
    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))
    fig.clf()


################ plot_oa_simulation_1


def plot_oa_simulation_1_timestep(scene, scene_df, actual_trajectory, frame_idx,
        planned_frame, planned_trajectory, planned_controls, road_segs, ego_bbox,
        step_horizon, filename="oa_simulation", road_boundary_constraints=True):
    """Helper function of plot_oa_simulation_1()
    """

    fig, ax = plt.subplots(figsize=(6, 6))
    render_scene(ax, scene, global_coordinates=True)
    if road_boundary_constraints:
        for A, b in road_segs.polytopes:
            util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.2)

    ovehicle_colors = get_ovehicle_color_set()
    frame_df = scene_df.loc[scene_df['frame_id'] == frame_idx]
    frame_df = frame_df.loc[frame_df['node_id'] != 'ego']
    for idx, (_, node_s) in enumerate(frame_df.iterrows()):
        """Plot OV positions"""
        position = node_s[['position_x', 'position_y']].values
        heading = node_s['heading_°']
        # lw = node_s[['length', 'width']].values # no bounding box in dataset
        lw = np.array([3.70, 1.79])
        vertices = util.vertices_from_bbox(position, heading, lw)
        color = ovehicle_colors[idx][0]
        bb = patches.Polygon(vertices.reshape((-1,2,)),
                closed=True, color=color, fc=util.plu.modify_alpha(color, 0.2))
        ax.add_patch(bb)
    
    "EV bounding box at current position"
    position = actual_trajectory[frame_idx, :2]
    heading = actual_trajectory[frame_idx, 2]
    vertices = util.vertices_from_bbox(position, heading, ego_bbox)
    bb = patches.Polygon(vertices.reshape((-1,2,)),
            closed=True, color="k", fc=util.plu.modify_alpha("black", 0.2))
    ax.add_patch(bb)

    "Plot EV planned trajectory"
    planned_xy = planned_trajectory[:, :2]
    ax.plot(planned_xy.T[0], planned_xy.T[1], "--bo", zorder=20, markersize=2)

    "Plot EV actual trajectory"
    # plans are made in current frame, and carried out next frame
    actual_xy = actual_trajectory[:, :2]
    idx = frame_idx + 1
    ax.plot(actual_xy[:idx, 0], actual_xy[:idx, 1], '-ko')
    ax.plot(actual_xy[idx:(idx + step_horizon), 0], actual_xy[idx:(idx + step_horizon), 1], '-o', color="orange")
    ax.plot(actual_xy[(idx + step_horizon):, 0], actual_xy[(idx + step_horizon):, 1], '-ko')

    "Configure plot"
    min_x, min_y = np.min(planned_xy, axis=0) - 20
    max_x, max_y = np.max(planned_xy, axis=0) + 20
    x_mid, y_mid = (max_x + min_x) / 2, (max_y + min_y) / 2
    extent = (x_mid - 30, x_mid + 30, y_mid - 30, y_mid + 30)
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"frame {frame_idx + 1}")
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}_idx{frame_idx}.png"))
    fig.clf()

def plot_oa_simulation_1(scene, actual_trajectory, planned_trajectories,
        planned_controls, road_segs, ego_bbox, step_horizon, step_size,
        filename="oa_simulation", road_boundary_constraints=True):
    """Plot to compare between actual trajectory and planned trajectories.
    Produces one plot per MPC step.
    
    Parameters
    ==========
    scene : Scene
        A scene produced by SceneBuilder containing environment and OV data.
    actual_trajectory : collections.OrderedDict of (int, ndarray)
        Indexed by frame ID. The EV's coordinate information produced by
        carlautil.actor_to_Lxyz_Vxyz_Axyz_Blwh_Rpyr_ndarray().
    planned_trajectories : collections.OrderedDict of (int, ndarray)
        Indexed by frame ID. The EV's planned state over timesteps T with
        (x position, y position, heading, speed, steering angle) as ndarray
        of shape (T + 1, 5). Includes origin.
    planned_controls : 
        Indexed by frame ID. The EV's planned controls over timesteps T with
        (acceleration, steering rate) as ndarray of shape (T, 2).
    road_segs : util.AttrDict
        Container of road segment properties.
    ego_bbox : ndarray
        The length and width of EV.
    step_horizon : int
        Number of steps to take at each iteration of MPC.
    """
    scene_df = scene_to_df(scene)
    scene_df[['position_x', 'position_y']] += np.array([scene.x_min, scene.y_min])
    scene_df = scene_df.sort_values('node_id')

    frames, actual_trajectory = util.unzip([i for i in actual_trajectory.items()])
    frames = np.array(frames)
    actual_trajectory = np.stack(actual_trajectory)
    planned_frames, planned_controls = util.unzip([i for i in planned_controls.items()])
    planned_frames, planned_trajectories = util.unzip([i for i in planned_trajectories.items()])
    
    for planned_frame, planned_trajectory in zip(planned_frames, planned_trajectories, planned_controls):
        frame_idx = np.argwhere(frames == planned_frame)[0, 0]
        plot_oa_simulation_1_timestep(scene, scene_df, actual_trajectory, frame_idx,
                planned_frame, planned_trajectory, planned_controls, road_segs, ego_bbox,
                step_horizon,
                filename=filename, road_boundary_constraints=road_boundary_constraints)

################ plot_oa_simulation_2

def plot_oa_simulation_2_timestep(scene, scene_df, actual_trajectory, frame_idx,
        planned_frame, planned_trajectory, planned_control, road_segs, ego_bbox,
        step_horizon, step_size, n_frames, filename="oa_simulation",
        road_boundary_constraints=True):
    """Helper function of plot_oa_simulation_2()
    """

    # Generate plots for map, velocity, heading and steering
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    ax = axes[0]
    render_scene(ax, scene, global_coordinates=True)
    if road_boundary_constraints:
        for A, b in road_segs.polytopes:
            util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.2)

    ovehicle_colors = get_ovehicle_color_set()
    frame_df = scene_df.loc[scene_df['frame_id'] == frame_idx]
    frame_df = frame_df.loc[frame_df['node_id'] != 'ego']
    for idx, (_, node_s) in enumerate(frame_df.iterrows()):
        """Plot OV positions"""
        position = node_s[['position_x', 'position_y']].values
        heading = node_s['heading_°']
        # lw = node_s[['length', 'width']].values # no bounding box in dataset
        lw = np.array([3.70, 1.79])
        vertices = util.vertices_from_bbox(position, heading, lw)
        color = ovehicle_colors[idx][0]
        bb = patches.Polygon(vertices.reshape((-1,2,)),
                closed=True, color=color, fc=util.plu.modify_alpha(color, 0.2))
        ax.add_patch(bb)
    
    "EV bounding box at current position"
    position = actual_trajectory[frame_idx, :2]
    heading = actual_trajectory[frame_idx, 2]
    vertices = util.vertices_from_bbox(position, heading, ego_bbox)
    bb = patches.Polygon(vertices.reshape((-1,2,)),
            closed=True, color="k", fc=util.plu.modify_alpha("black", 0.2))
    ax.add_patch(bb)

    """Plot EV planned trajectory.
    This is computed from CPLEX using LTI vehicle dynamics."""
    planned_xy = planned_trajectory[:, :2]
    ax.plot(planned_xy.T[0], planned_xy.T[1], "--bo", zorder=20, markersize=2)

    """Plot EV planned trajectory.
    This is with original non-linear dyamics and controls from CPLEX."""
    gt_planned_trajectory = compute_nonlinear_dynamical_states(planned_trajectory[0],
            planned_trajectory.shape[0] - 1, step_size, planned_control, l_r=0.5*ego_bbox[0], L=ego_bbox[0])
    gt_planned_xy = gt_planned_trajectory[:, :2]
    ax.plot(*gt_planned_xy.T, "--go", zorder=20, markersize=2)

    "Plot EV actual trajectory"
    # plans are made in current frame, and carried out next frame
    actual_xy = actual_trajectory[:, :2]
    idx = frame_idx + 1
    ax.plot(actual_xy[:idx, 0], actual_xy[:idx, 1], '-ko')
    ax.plot(actual_xy[idx:(idx + step_horizon), 0], actual_xy[idx:(idx + step_horizon), 1], '-o', color="orange")
    ax.plot(actual_xy[(idx + step_horizon):, 0], actual_xy[(idx + step_horizon):, 1], '-ko')

    "Configure plot"
    min_x, min_y = np.min(planned_xy, axis=0) - 20
    max_x, max_y = np.max(planned_xy, axis=0) + 20
    x_mid, y_mid = (max_x + min_x) / 2, (max_y + min_y) / 2
    extent = (x_mid - 30, x_mid + 30, y_mid - 30, y_mid + 30)
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"frame {frame_idx + 1}")
    ax.set_aspect('equal')

    """Plot v, which is the vehicle speed"""
    ax = axes[1]
    planned_v = planned_trajectory[:, 3]
    gt_planned_v = gt_planned_trajectory[:, 3]
    actual_v  = actual_trajectory[:, 3]
    ax.plot(range(1, n_frames + 1), actual_v, "-k.", label="ground truth")
    ax.plot(range(idx, idx + planned_v.size), planned_v, "-b.", label="under LTI")
    ax.plot(range(idx, idx + gt_planned_v.size), gt_planned_v, "-g.", label="without LTI")
    ax.set_title("$v$ speed of c.g., m/s")
    ax.set_ylabel("m/s")

    """Plot psi, which is the vehicle longitudinal angle in global coordinates"""
    ax = axes[2]
    planned_psi = planned_trajectory[:, 2]
    gt_planned_psi = gt_planned_trajectory[:, 2]
    actual_psi  = actual_trajectory[:, 2]
    ax.plot(range(1, n_frames + 1), actual_psi, "-k.", label="ground truth")
    ax.plot(range(idx, idx + planned_psi.size), planned_psi, "-b.", label="under LTI")
    ax.plot(range(idx, idx + gt_planned_psi.size), gt_planned_psi, "-g.", label="without LTI")
    ax.set_title("$\psi$ longitudinal angle, radians")
    ax.set_ylabel("rad")

    """Plot delta, which is the turning angle"""
    ax = axes[3]
    planned_delta = planned_trajectory[:, 4]
    gt_planned_delta = gt_planned_trajectory[:, 4]
    actual_delta  = actual_trajectory[:, 4]
    ax.plot(range(1, n_frames + 1), actual_delta, "-k.", label="under LTI")
    ax.plot(range(idx, idx + planned_delta.size), planned_delta, "-b.", label="under LTI")
    ax.plot(range(idx, idx + gt_planned_delta.size), gt_planned_delta, "-g.", label="without LTI")
    ax.set_title("$\delta$ turning angle, radians")
    ax.set_ylabel("rad")

    """Plot a, which is acceleration control input"""
    ax = axes[4]
    planned_a = planned_control[:, 0]
    ax.plot(range(idx, idx + planned_a.size), planned_a, "-.", color="orange")
    ax.set_title("$a$ acceleration input $m/s^2$")

    ax = axes[5]
    planned_ddelta = planned_control[:, 1]
    ax.plot(range(idx, idx + planned_ddelta.size), planned_ddelta, "-.", color="orange")
    ax.set_title("$\dot{\delta}$ turning rate input rad/s")

    for ax in axes[1:4]:
        ax.set_xlabel("time, s")
        ax.grid()
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}_idx{frame_idx}.png"))
    fig.clf()


def plot_oa_simulation_2(scene, actual_trajectory, planned_trajectories,
        planned_controls, road_segs, ego_bbox, step_horizon, step_size,
        filename="oa_simulation", road_boundary_constraints=True):
    """Plot to compare between actual trajectory and planned trajectories.
    Produces one plot per MPC step. Produces plots for velocity, steering.
    
    Parameters
    ==========
    scene : Scene
        A scene produced by SceneBuilder containing environment and OV data.
    actual_trajectory : collections.OrderedDict of (int, ndarray)
        Indexed by frame ID. The EV's coordinate information produced by
        carlautil.actor_to_Lxyz_Vxyz_Axyz_Blwh_Rpyr_ndarray().
    planned_trajectories : collections.OrderedDict of (int, ndarray)
        Indexed by frame ID. The EV's planned state over timesteps T with
        (x position, y position, heading, speed, steering angle) as ndarray
        of shape (5, T + 1).
    planned_controls : 
        Indexed by frame ID. The EV's planned controls over timesteps T with
        (acceleration, steering rate) as ndarray of shape (2, T).
    road_segs : util.AttrDict
        Container of road segment properties.
    """
    scene_df = scene_to_df(scene)
    scene_df[['position_x', 'position_y']] += np.array([scene.x_min, scene.y_min])
    scene_df = scene_df.sort_values('node_id')

    frames, actual_trajectory = util.unzip([i for i in actual_trajectory.items()])
    frames = np.array(frames)
    actual_trajectory = np.stack(actual_trajectory)
    planned_frames, planned_controls = util.unzip([i for i in planned_controls.items()])
    planned_frames, planned_trajectories = util.unzip([i for i in planned_trajectories.items()])
    
    for planned_frame, planned_trajectory, planned_control \
            in zip(planned_frames, planned_trajectories, planned_controls):
        frame_idx = np.argwhere(frames == planned_frame)[0, 0]
        plot_oa_simulation_2_timestep(scene, scene_df, actual_trajectory, frame_idx,
                planned_frame, planned_trajectory, planned_control, road_segs, ego_bbox,
                step_horizon, step_size, frames.size,
                filename=filename, road_boundary_constraints=road_boundary_constraints)


################ plot_oa_simulation

def plot_oa_simulation_0(scene, actual_trajectory, planned_trajectories,
        planned_controls, road_segs, ego_bbox, step_horizon, step_size,
        filename="oa_simulation", road_boundary_constraints=True):
    """Plot to compare between actual trajectory and planned trajectories.
    This method puts all MPC steps in one plot."""

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
    if axes.ndim == 1:
        axes = axes[None]

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

##
# TODO: clear above, restart

# plot_oa_simulation = plot_oa_simulation_0

def plot_oa_simulation_timestep(
        scene, scene_df, node_ids, actual_trajectory, frame_idx,
        planned_frame, planned_trajectory, planned_control, road_segs, ego_bbox,
        step_horizon, step_size, n_frames, filename="oa_simulation",
        road_boundary_constraints=True):
    """Helper function of plot_oa_simulation_2()"""

    # Generate plots for map, velocity, heading and steering
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    ax = axes[0]
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
    
    "EV bounding box at current position"
    position = actual_trajectory[frame_idx, :2]
    heading = actual_trajectory[frame_idx, 2]
    vertices = util.vertices_from_bbox(position, heading, ego_bbox)
    bb = patches.Polygon(vertices.reshape((-1,2,)),
            closed=True, color="k", fc=util.plu.modify_alpha("black", 0.2))
    ax.add_patch(bb)

    """Plot EV planned trajectory.
    This is computed from CPLEX using LTI vehicle dynamics."""
    planned_xy = planned_trajectory[:, :2]
    ax.plot(planned_xy.T[0], planned_xy.T[1], "--bo", zorder=20, markersize=2)

    """Plot EV planned trajectory.
    This is with original non-linear dyamics and controls from CPLEX."""
    gt_planned_trajectory = compute_nonlinear_dynamical_states(planned_trajectory[0],
            planned_trajectory.shape[0] - 1, step_size, planned_control, l_r=0.5*ego_bbox[0], L=ego_bbox[0])
    gt_planned_xy = gt_planned_trajectory[:, :2]
    ax.plot(*gt_planned_xy.T, "--go", zorder=20, markersize=2)

    "Plot EV actual trajectory"
    # plans are made in current frame, and carried out next frame
    actual_xy = actual_trajectory[:, :2]
    idx = frame_idx + 1
    ax.plot(actual_xy[:idx, 0], actual_xy[:idx, 1], '-ko')
    ax.plot(actual_xy[idx:(idx + step_horizon), 0], actual_xy[idx:(idx + step_horizon), 1], '-o', color="orange")
    ax.plot(actual_xy[(idx + step_horizon):, 0], actual_xy[(idx + step_horizon):, 1], '-ko')

    "Configure plot"
    min_x, min_y = np.min(planned_xy, axis=0) - 20
    max_x, max_y = np.max(planned_xy, axis=0) + 20
    x_mid, y_mid = (max_x + min_x) / 2, (max_y + min_y) / 2
    extent = (x_mid - 30, x_mid + 30, y_mid - 30, y_mid + 30)
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"frame {frame_idx + 1}")
    ax.set_aspect('equal')

    """Plot v, which is the vehicle speed"""
    ax = axes[1]
    planned_v = planned_trajectory[:, 3]
    gt_planned_v = gt_planned_trajectory[:, 3]
    actual_v  = actual_trajectory[:, 3]
    ax.plot(range(1, n_frames + 1), actual_v, "-k.", label="ground truth")
    ax.plot(range(idx, idx + planned_v.size), planned_v, "-b.", label="under LTI")
    ax.plot(range(idx, idx + gt_planned_v.size), gt_planned_v, "-g.", label="without LTI")
    ax.set_title("$v$ speed of c.g., m/s")
    ax.set_ylabel("m/s")

    """Plot psi, which is the vehicle longitudinal angle in global coordinates"""
    ax = axes[2]
    planned_psi = planned_trajectory[:, 2]
    gt_planned_psi = gt_planned_trajectory[:, 2]
    actual_psi  = actual_trajectory[:, 2]
    ax.plot(range(1, n_frames + 1), actual_psi, "-k.", label="ground truth")
    ax.plot(range(idx, idx + planned_psi.size), planned_psi, "-b.", label="under LTI")
    ax.plot(range(idx, idx + gt_planned_psi.size), gt_planned_psi, "-g.", label="without LTI")
    ax.set_title("$\psi$ longitudinal angle, radians")
    ax.set_ylabel("rad")

    """Plot delta, which is the turning angle"""
    ax = axes[3]
    planned_delta = planned_trajectory[:, 4]
    gt_planned_delta = gt_planned_trajectory[:, 4]
    actual_delta  = actual_trajectory[:, 4]
    ax.plot(range(1, n_frames + 1), actual_delta, "-k.", label="under LTI")
    ax.plot(range(idx, idx + planned_delta.size), planned_delta, "-b.", label="under LTI")
    ax.plot(range(idx, idx + gt_planned_delta.size), gt_planned_delta, "-g.", label="without LTI")
    ax.set_title("$\delta$ turning angle, radians")
    ax.set_ylabel("rad")

    """Plot a, which is acceleration control input"""
    ax = axes[4]
    planned_a = planned_control[:, 0]
    ax.plot(range(idx, idx + planned_a.size), planned_a, "-.", color="orange")
    ax.set_title("$a$ acceleration input $m/s^2$")

    ax = axes[5]
    planned_ddelta = planned_control[:, 1]
    ax.plot(range(idx, idx + planned_ddelta.size), planned_ddelta, "-.", color="orange")
    ax.set_title("$\dot{\delta}$ turning rate input rad/s")

    for ax in axes[1:4]:
        ax.set_xlabel("time, s")
        ax.grid()
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}_idx{frame_idx}.png"))
    fig.clf()

def plot_oa_simulation(
    scene, map_data, actual_trajectory, planned_trajectories, planned_controls, goals,
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
    lowlevel : util.AttrDict
        PID statistics.
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
    scene_df = scene_to_df(scene)
    scene_df[['position_x', 'position_y']] += np.array([scene.x_min, scene.y_min])
    node_ids = scene_df['node_id'].unique()
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
            scene, scene_df, node_ids, map_data, actual_trajectory, frame_idx, planned_frame,
            planned_trajectory, planned_control, goal, road_segs, ego_bbox, step_horizon,
            steptime, frames.size, filename=filename,
            road_boundary_constraints=road_boundary_constraints
        )
