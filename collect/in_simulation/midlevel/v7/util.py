"""DEPRECATED unused(?) plotting functions"""

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
from ..util import get_ovehicle_color_set, get_vertices_from_center
from ....visualize.trajectron import render_scene, render_map_crop
from ....trajectron import scene_to_df

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
PADDING = 16


def plot_multiple_coinciding_controls_timestep(
    ax, pred_result, ovehicles, params, ctrl_result,
    X_star, t, traj_idx, latent_indices,
    ego_bbox, extent=None
):
    """Helper function for plot_multiple_coinciding_controls()
    
    Parameters
    ==========
    ax : matplotlib.axes.Axes
        The plot to make.
    pred_result : util.AttrDict
        Prediction payload containing: scene, timestep, nodes,
        predictions, z, latent_probs, past_dict, ground_truth_dict.
    ovehicles : list of OVehicle
        Vehicles present in the scene.
    params : util.AttrDict
        Parameters of optimization.
    ctrl_result : util.AttrDict
        Control optimization payload containing: cost, U_star,
        X_star, goal, A_unions, b_unions, vertices.
    X_star : ndarray
        The predicted states including initial state from optimization.
    t : int
        The timestep of predicted state we want to show for this plot.
        The state corresponding to timestep 1 is X_star[t + 1].
    traj_idx : int
    latent_indices : ndarray of int
    ego_bbox : list of int
        Ego bounding box (longitudinal, lateral) dimensions.
    extent : tuple of number
        Extent of plot (x min, x max, y min, y max).
    """
    ovehicle_colors = get_ovehicle_color_set()
    render_scene(ax, pred_result.scene, global_coordinates=True)
    ax.plot(
        ctrl_result.goal[0], ctrl_result.goal[1],
        marker='*', markersize=8, color="yellow"
    )

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
        ego_bbox
    )
    bb = patches.Polygon(
        vertices.reshape((-1,2,)), closed=True, color='k',
        fc=util.plu.modify_alpha("black", 0.2), ls='-'
    )
    ax.add_patch(bb)

    # Plot ego vehicle planned trajectory
    ax.plot(X_star[:(t + 2), 0], X_star[:(t + 2), 1], 'k-o', markersize=2)

    # Get vertices of EV and plot its bounding box
    if t < 2:
        vertices = util.vertices_from_bbox(
            X_star[t + 1, :2], X_star[t + 1, 2], ego_bbox
        )
    else:
        # Make EV bbox look better on far horizons by fixing the heading angle
        heading = np.arctan2(
            X_star[t + 1, 1] - X_star[t, 1], X_star[t + 1, 0] - X_star[t, 0]
        )
        vertices = util.vertices_from_bbox(X_star[t + 1, :2], heading, ego_bbox)
    
    bb = patches.Polygon(
        vertices.reshape((-1,2,)), closed=True, color='k', fc='none', ls='-'
    )
    ax.add_patch(bb)

    # Plot other vehicles
    for ov_idx, ovehicle in enumerate(ovehicles):
        # Plot past trajectory
        latent_idx = latent_indices[ov_idx]
        color = ovehicle_colors[ov_idx][0]
        ax.plot(ovehicle.past[:,0], ovehicle.past[:,1],
                marker='o', markersize=2, color=color)

        # Plot overapproximation
        color = ovehicle_colors[ov_idx][latent_idx]
        A = ctrl_result.A_unions[traj_idx][t][ov_idx]
        b = ctrl_result.b_unions[traj_idx][t][ov_idx]
        try:
            util.npu.plot_h_polyhedron(ax, A, b, ec=color, alpha=1)
        except scipy.spatial.qhull.QhullError as e:
            print(f"Failed to plot polyhedron at timestep t={t}")
            
        # Plot vertices
        vertices = ctrl_result.vertices[t][latent_idx][ov_idx]
        X = vertices[:, 0:2].T
        ax.scatter(X[0], X[1], color=color, s=2)
        X = vertices[:, 2:4].T
        ax.scatter(X[0], X[1], color=color, s=2)
        X = vertices[:, 4:6].T
        ax.scatter(X[0], X[1], color=color, s=2)
        X = vertices[:, 6:8].T
        ax.scatter(X[0], X[1], color=color, s=2)

    if extent is not None:
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"t = {t + 1}")
    ax.set_aspect('equal')


def plot_multiple_coinciding_controls(
    pred_result, ovehicles, params, ctrl_result, T,
    ego_bbox, filename='multiple_control'
):
    """Plot predictions and control trajectory for v7 control.

    Parameters
    ==========
    pred_result : util.AttrDict
        Prediction payload containing: scene, timestep, nodes,
        predictions, z, latent_probs, past_dict, ground_truth_dict.
    ovehicles : list of OVehicle
        Vehicles present in the scene.
    params : util.AttrDict
        Parameters of optimization.
    ctrl_result : util.AttrDict
        Control optimization payload containing: cost, U_star,
        X_star, goal, A_unions, b_unions, vertices.
    T : int
        Control horizon.
    ego_bbox : list of int
        Ego bounding box (longitudinal, lateral) dimensions.
    filename : str
        The filename to save.
    """

    """Get control trajectory data"""
    X_init = np.repeat(params.initial_state.world[None], params.N_select, axis=0)
    X_star = np.concatenate((X_init[:, None], ctrl_result.X_star), axis=1)
    _X_star = X_star[..., :2].reshape(-1, 2)
    x_min, y_min = np.min(_X_star, axis=0)
    x_max, y_max = np.max(_X_star, axis=0)
    x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
    extent = (x_mid - PADDING, x_mid + PADDING, y_mid - PADDING, y_mid + PADDING)
    
    for traj_idx in range(params.N_select):
        """Generate a single plot for each combination of overapproximations
        that we have applied control over."""
        fig, axes = plt.subplots(T // 2 + (T % 2), 2, figsize=(10, (10 / 4)*T))
        axes = axes.ravel()
        latent_indices = params.sublist_joint_decisions[traj_idx]
        for t, ax in enumerate(axes):
            plot_multiple_coinciding_controls_timestep(
                ax, pred_result, ovehicles, params, ctrl_result,
                X_star[traj_idx], t, traj_idx, latent_indices,
                ego_bbox, extent=extent
            )

        fig.tight_layout()
        fig.savefig(os.path.join('out', f"{filename}_traj{traj_idx + 1}.png"))
        fig.clf()


def plot_oa_simulation_timestep(
    scene, scene_df, node_ids, map_data, actual_trajectory, frame_idx,
    planned_frame, planned_trajectory, planned_control, goal, road_segs,
    ego_bbox, step_horizon, n_coincide, steptime, n_frames,
    filename="oa_simulation", road_boundary_constraints=True
):
    """Helper function of plot_oa_simulation()"""

    # coinciding component of trajectory
    coin_trajectory = planned_trajectory[0, :(n_coincide + 1)]
    coin_control = planned_control[0, :n_coincide]
    coin_xy = coin_trajectory[:, :2]
    coin_psi = coin_trajectory[:, 2]
    coin_v = coin_trajectory[:, 3]
    coin_a = coin_control[:, 0]
    coin_delta = coin_control[:, 1]

    # contingency component of trajectory
    cont_trajectory = planned_trajectory[:, n_coincide:]
    cont_control = planned_control[:, (n_coincide - 1):]
    cont_xy = cont_trajectory[..., :2]
    cont_psi = cont_trajectory[..., 2]
    cont_v = cont_trajectory[..., 3]
    cont_a  = cont_control[..., 0]
    cont_delta = cont_control[..., 1]

    n_select = planned_trajectory.shape[0]
    actual_xy = actual_trajectory[..., :2]
    actual_psi  = actual_trajectory[..., 2]
    actual_v  = actual_trajectory[..., 3]
    _planned_xy = planned_trajectory[..., :2].reshape(-1, 2)
    min_x, min_y = np.min(_planned_xy, axis=0)
    max_x, max_y = np.max(_planned_xy, axis=0)
    x_mid, y_mid = (max_x + min_x) / 2, (max_y + min_y) / 2
    extent = (x_mid - PADDING, x_mid + PADDING, y_mid - PADDING, y_mid + PADDING)

    # Generate plots for map, state and inputs
    traj_colors = cm.winter(np.linspace(0, 1, n_select))
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.ravel()
    ax = axes[0]

    """Map overlay"""
    render_map_crop(ax, map_data, extent)
    if road_boundary_constraints:
        for A, b in road_segs.polytopes:
            util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.2)
    ax.plot(goal.x, goal.y, marker='*', markersize=8, color="yellow")

    for idx, node_id in enumerate(node_ids):
        """Plot OV trajectory"""
        if node_id == "ego": continue
        node_df = scene_df[scene_df['node_id'] == node_id]
        X = node_df[['position_x', 'position_y']].values.T
        ax.plot(X[0], X[1], ':.', color=AGENT_COLORS[idx % NCOLORS])

    "EV bounding box at current position"
    position = actual_trajectory[frame_idx, :2]
    heading = actual_trajectory[frame_idx, 2]
    vertices = util.npu.vertices_from_bbox(position, heading, ego_bbox)
    bb = patches.Polygon(
        vertices.reshape((-1,2,)), closed=True, color="k",
        fc=util.plu.modify_alpha("black", 0.2)
    )
    ax.add_patch(bb)

    """Plot EV planned contingency trajectory. This is computed
    from CPLEX using linearized discrete-time vehicle dynamics."""
    ax.plot(*coin_xy.T, "--bo", zorder=20, markersize=2)
    for traj_idx in range(n_select):
        # plot contingency plans separately
        ax.plot(
            *cont_xy[traj_idx].T, "--o",
            color=traj_colors[traj_idx], zorder=20, markersize=2
        )
    
    """Plot EV actual trajectory.
    Plans are made in current frame, and carried out next frame."""
    idx = frame_idx + 1
    ax.plot(*actual_xy[:idx].T, '-ko')
    ax.plot(*actual_xy[idx:(idx + step_horizon)].T, '-o', color="orange")
    ax.plot(*actual_xy[(idx + step_horizon):].T, '-ko')        

    """Plot bounding boxes along with trajectories on xy-plane."""
    ax = axes[1]
    ax.plot(*coin_xy.T, "b.", zorder=20, markersize=2)
    ax.plot(*actual_xy.T, 'ko')
    vertices = util.npu.vertices_of_bboxes(actual_xy, actual_psi, ego_bbox)
    for v in vertices:
        bb = patches.Polygon(v, closed=True, color="k", fc="none")
        ax.add_patch(bb)
    vertices = util.npu.vertices_of_bboxes(coin_xy, coin_psi, ego_bbox)
    for v in vertices:
        bb = patches.Polygon(
            v, closed=True, color="b", fc=util.plu.modify_alpha("blue", 0.2)
        )
        ax.add_patch(bb)
    for traj_idx in range(n_select):
        # plot contingency plans separately
        ax.plot(
            *cont_xy[traj_idx].T, ".",
            color=traj_colors[traj_idx], zorder=20, markersize=2
        )
        vertices = util.npu.vertices_of_bboxes(
            cont_xy[traj_idx], cont_psi[traj_idx], ego_bbox
        )
        for v in vertices:
            bb = patches.Polygon(
                v, closed=True, color=traj_colors[traj_idx],
                fc=util.plu.modify_alpha(traj_colors[traj_idx], 0.2)
            )
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
    ax.plot(range(idx, idx + coin_v.size), coin_v, "-b.", label="linear plan")
    for traj_idx in range(n_select):
        ax.plot(
            range(idx + n_coincide, idx + n_coincide + cont_v[traj_idx].size),
            cont_v[traj_idx], "-.", color=traj_colors[traj_idx]
        )
    ax.set_title("$v$ speed of c.g., m/s")
    ax.set_ylabel("m/s")

    """Plot psi, which is the vehicle longitudinal angle in global coordinates."""
    ax = axes[3]
    ax.plot(range(1, n_frames + 1), actual_psi, "-k.", label="actual")
    ax.plot(range(idx, idx + coin_psi.size), coin_psi, "-b.", label="linear plan")
    for traj_idx in range(n_select):
        ax.plot(
            range(idx + n_coincide, idx + n_coincide + cont_psi[traj_idx].size),
            cont_psi[traj_idx], "-.", color=traj_colors[traj_idx]
        )
    ax.set_title("$\psi$ longitudinal angle, radians")
    ax.set_ylabel("rad")

    """Plot a, which is acceleration control input"""
    ax = axes[4]
    ax.plot(range(idx, idx + coin_a.size), coin_a, "-b.", label="control plan")
    for traj_idx in range(n_select):
        ax.plot(
            range(idx + n_coincide - 1, idx + n_coincide - 1 + cont_a[traj_idx].size),
            cont_a[traj_idx], "-.", color=traj_colors[traj_idx]
        )
    ax.set_title("$a$ acceleration input, $m/s^2$")

    """Plot delta, which is steering control input"""
    ax = axes[5]
    ax.plot(range(idx, idx + coin_delta.size), coin_delta, "-b.", label="control plan")
    for traj_idx in range(n_select):
        ax.plot(
            range(idx + n_coincide - 1, idx + n_coincide - 1 + cont_delta[traj_idx].size),
            cont_delta[traj_idx], "-.", color=traj_colors[traj_idx]
        )
    ax.set_title("$\delta$ steering input, radians")

    for ax in axes[2:]:
        ax.set_xlabel("timestep")
        ax.grid()
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}_idx{frame_idx}.png"))
    fig.clf()


def plot_oa_simulation(
    scene, map_data, actual_trajectory, planned_trajectories, planned_controls, goals,
    lowlevel, road_segs, ego_bbox, step_horizon, n_coincide, steptime,
    filename="oa_simulation", road_boundary_constraints=True
):
    """Plot to compare between actual trajectory and planned trajectories.
    Produces one plot per MPC step. Produces plots for velocity, steering.

    Parameters
    ==========
    planned_trajectories : collections.OrderedDict of (int, ndarray)
        Indexed by frame ID. The EV's planned contingency states over timesteps T+1
        incl. origin with global (x, y) position, heading and speed as ndarray of
        shape (*, T + 1, 4). The first dimension depends on the OV predicitons.
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
            scene, scene_df, node_ids, map_data, actual_trajectory, frame_idx,
            planned_frame, planned_trajectory, planned_control, goal, road_segs,
            ego_bbox, step_horizon, n_coincide, steptime, frames.size,
            filename=filename, road_boundary_constraints=road_boundary_constraints
        )


