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
from ...visualize.trajectron import render_scene
from ...trajectron import scene_to_df

# Profiling libraries
import functools
import cProfile, pstats, io

AGENT_COLORS = [
        'blue', 'darkviolet', 'dodgerblue', 'darkturquoise',
        'green', 'gold', 'orange', 'red', 'deeppink']
AGENT_COLORS = np.array(AGENT_COLORS) \
        .take([(i * 5) % len(AGENT_COLORS) for i in range(17)], 0)
NCOLORS = len(AGENT_COLORS)

def profile(sort_by='cumulative', lines_to_print=None, strip_dirs=False):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/
    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            if strip_dirs:
                ps.strip_dirs()
            ps.sort_stats(sort_by)
            ps.print_stats(lines_to_print)
            logging.info(f"code profile of {func.__name__}")
            logging.info(s.getvalue())
            return retval
        return wrapper
    return inner

def get_vertices_from_center(center, heading, lw):
    """Compute the verticles of a vehicle 
    """
    vertices = np.empty((8,))
    rot1 = np.array([
            [ np.cos(heading),  np.sin(heading)],
            [ np.sin(heading), -np.cos(heading)]])
    rot2 = np.array([
            [ np.cos(heading), -np.sin(heading)],
            [ np.sin(heading),  np.cos(heading)]])
    rot3 = np.array([
            [-np.cos(heading), -np.sin(heading)],
            [-np.sin(heading),  np.cos(heading)]])
    rot4 = np.array([
            [-np.cos(heading),  np.sin(heading)],
            [-np.sin(heading), -np.cos(heading)]])
    vertices[0:2] = center + 0.5 * rot1 @ lw
    vertices[2:4] = center + 0.5 * rot2 @ lw
    vertices[4:6] = center + 0.5 * rot3 @ lw
    vertices[6:8] = center + 0.5 * rot4 @ lw
    return vertices

def get_vertices_from_centers(centers, headings, lw):
    """Like get_vertices_from_center() but broadcasted
    over multiple centers and headings.
    """
    C = np.cos(headings)
    S = np.sin(headings)
    rot11 = np.stack(( C,  S), axis=-1)
    rot12 = np.stack(( S, -C), axis=-1)
    rot21 = np.stack(( C, -S), axis=-1) 
    rot22 = np.stack(( S,  C), axis=-1)
    rot31 = np.stack((-C, -S), axis=-1)
    rot32 = np.stack((-S,  C), axis=-1)
    rot41 = np.stack((-C,  S), axis=-1)
    rot42 = np.stack((-S, -C), axis=-1)
    # Rot has shape (1000, 8, 2)
    Rot = np.stack((rot11, rot12, rot21, rot22, rot31, rot32, rot41, rot42), axis=1)
    # disp has shape (1000, 8)
    disp = 0.5 * Rot @ lw
    # centers has shape (1000, 8)
    centers = np.tile(centers, (4,))
    return centers + disp

def obj_matmul(A, B):
    """Non-vectorized multiplication of arrays of object dtype"""
    if len(B.shape) == 1:
        C = np.zeros((A.shape[0]), dtype=object)
        for i in range(A.shape[0]):
            for k in range(A.shape[1]):
                C[i] += A[i,k]*B[k]
    else:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=object)
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    C[i,j] += A[i,k]*B[k,j]
    return C

def get_approx_union(theta, vertices):
    """Gets (A, b) for the contraint set A x >= b.
    TODO: deprecate. Use updated compute_outerapproximation()

    Parameters
    ==========
    theta : float
        Mean angle of 
    vertices : np.array
        Vertices of shape (N, 8)
    
    Returns
    =======
    np.array
        A matrix of shape (4, 2)
    np.array
        b vector of shape (4,)
    """
    At = np.array([
            [ np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])
    At = np.concatenate((np.eye(2), -np.eye(2),)) @ At

    a0 = np.max(At @ vertices[:, 0:2].T, axis=1)
    a1 = np.max(At @ vertices[:, 2:4].T, axis=1)
    a2 = np.max(At @ vertices[:, 4:6].T, axis=1)
    a3 = np.max(At @ vertices[:, 6:8].T, axis=1)
    b0 = np.max(np.stack((a0, a1, a2, a3)), axis=0)
    return At, b0

def compute_L4_outerapproximation(theta, vertices):
    """Gets outerapproximation (A, b) with L=4
    sides containing bounding box vertices.
    Outerapproximation forms the contraints A x >= b.

    Parameters
    ==========
    theta : float
        Mean heading of bounding boxes.
    vertices : np.array
        Vertices of bounding boxes with shape (N, 4, 2).
    
    Returns
    =======
    np.array
        A matrix of shape (4, 2)
    np.array
        b vector of shape (4,)
    """
    At = np.array([
            [ np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])
    At = np.concatenate((np.eye(2), -np.eye(2),)) @ At

    a0 = np.max(At @ vertices[:, 0].T, axis=1)
    a1 = np.max(At @ vertices[:, 1].T, axis=1)
    a2 = np.max(At @ vertices[:, 2].T, axis=1)
    a3 = np.max(At @ vertices[:, 3].T, axis=1)
    b0 = np.max(np.stack((a0, a1, a2, a3)), axis=0)
    return At, b0

def plot_h_polyhedron(ax, A, b, fc='none', ec='none', alpha=0.3):
    """
    A x < b is the H-representation
    [A; b], A x + b < 0 is the format for HalfspaceIntersection
    """
    Ab = np.concatenate((A, -b[...,None],), axis=-1)
    res = scipy.optimize.linprog([0, 0],
            A_ub=Ab[:,:2], b_ub=-Ab[:,2],
            bounds=(None, None))
    hs = scipy.spatial.HalfspaceIntersection(Ab, res.x)
    ch = scipy.spatial.ConvexHull(hs.intersections)
    x, y = zip(*hs.intersections[ch.vertices])
    ax.fill(x, y, fc=fc, ec=ec, alpha=alpha)

OVEHICLE_COLORS = [
    clr.LinearSegmentedColormap.from_list('ro', ['red', 'orange'], N=256),
    clr.LinearSegmentedColormap.from_list('gy', ['green', 'yellow'], N=256),
    clr.LinearSegmentedColormap.from_list('bp', ['blue', 'purple'], N=256),
    clr.LinearSegmentedColormap.from_list('td', ['turquoise', 'deeppink'], N=256),
    clr.LinearSegmentedColormap.from_list('bt', ['brown', 'teal'], N=256),
]

def get_ovehicle_color_set(latents=None):
    latents = [] if latents is None else latents
    ovehicle_colors = []
    for idx, ov_colormap in enumerate(OVEHICLE_COLORS):
        try:
            l = latents[idx]
        except IndexError:
            l = 5
        ov_colors = ov_colormap(np.linspace(0,1,l))
        ovehicle_colors.append(ov_colors)
    return ovehicle_colors

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
            ctrl_result.headings[t],
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
    X_star = np.concatenate((ctrl_result.start[None], ctrl_result.X_star[:, :2]))
    x_min, y_min = np.min(X_star, axis=0) - 20
    x_max, y_max = np.max(X_star, axis=0) + 20
    extent = (x_min, x_max, y_min, y_max)
    for t, ax in zip(range(T), axes):
        plot_lcss_prediction_timestep(ax, pred_result.scene, ovehicles,
                params, ctrl_result, X_star, t, ego_bbox, extent=extent)
    
    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))
    fig.clf()

def plot_oa_simulation(scene, actual_trajectory, planned_trajectories,
        planned_controls, road_segs, ego_bbox, step_horizon,
        filename="oa_simulation"):
    """Plot to compare between actual trajectory and planned trajectories"""

    scene_df = scene_to_df(scene)
    scene_df[['position_x', 'position_y']] += np.array([scene.x_min, scene.y_min])
    node_ids = scene_df['node_id'].unique()
    frames, actual_trajectory = util.unzip([i for i in actual_trajectory.items()])
    frames = np.array(frames)
    actual_trajectory = np.stack(actual_trajectory)
    actual_xy = actual_trajectory[:, :2]
    planned_frames, planned_trajectories = util.unzip([i for i in planned_trajectories.items()])
    
    N = len(planned_trajectories)
    fig, axes = plt.subplots(N // 2 + (N % 2), 2, figsize=(10, (10 / 4)*N))
    axes = axes.ravel()

    for ax in axes:
        """Render road overlay and plot actual trajectory"""
        render_scene(ax, scene, global_coordinates=True)
        for A, b in road_segs.polytopes:
            util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.2)
    
    for idx, node_id in enumerate(node_ids):
        """Plot OV trajectory"""
        if node_id == "ego": continue
        node_df = scene_df[scene_df['node_id'] == node_id]
        X = node_df[['position_x', 'position_y']].values.T
        for ax in axes:
            ax.plot(X[0], X[1], ':.', color=AGENT_COLORS[idx % NCOLORS])
    
    for ax, planned_frame, planned_trajectory in zip(axes, planned_frames, planned_trajectories):
        """Plot planned trajectories on separate plots"""
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

    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))
    fig.clf()

#########################################
# Methods for multiple coinciding control
#########################################

def _plot_multiple_coinciding_controls_timestep(ax, map_data, ovehicles,
        params, ctrl_result, X, headings, t, traj_idx, latent_indices, ego_bbox):
    ovehicle_colors = get_ovehicle_color_set()
    ax.imshow(map_data.road_bitmap, extent=map_data.extent,
            origin='lower', cmap=clr.ListedColormap(['none', 'grey']))
    ax.imshow(map_data.road_div_bitmap, extent=map_data.extent,
            origin='lower', cmap=clr.ListedColormap(['none', 'yellow']))
    ax.imshow(map_data.lane_div_bitmap, extent=map_data.extent,
            origin='lower', cmap=clr.ListedColormap(['none', 'silver']))

    ax.plot(ctrl_result.start[0], ctrl_result.start[1],
            marker='*', markersize=8, color="blue")
    ax.plot(ctrl_result.goal[0], ctrl_result.goal[1],
            marker='*', markersize=8, color="green")

    # Plot ego vehicle trajectory
    ax.plot(ctrl_result.X_star[traj_idx, :t, 0],
            ctrl_result.X_star[traj_idx, :t, 1], 'k-o', markersize=2)

    # Get vertices of EV and plot its bounding box
    vertices = get_vertices_from_center(
            ctrl_result.X_star[traj_idx, t, :2], headings[t], ego_bbox)
    bb = patches.Polygon(vertices.reshape((-1,2,)),
            closed=True, color='k', fc='none')
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

    ax.set_title(f"t = {t}")
    ax.set_aspect('equal')

def _plot_multiple_coinciding_controls(pred_result, ovehicles,
        params, ctrl_result, ego_bbox, filename='lcss_control'):
    """
    Parameters
    ==========
    predict_result : util.AttrDict
        Payload containing scene, timestep, nodes, predictions, z,
        latent_probs, past_dict, ground_truth_dict
    ovehicles : list of OVehicle
    params : util.AttrDict
    ctrl_result : util.AttrDict
    ego_bbox : list of int
    filename : str
    """

    """Plots for paper"""
    for traj_idx, latent_indices in enumerate(
            util.product_list_of_list([range(ovehicle.n_states) for ovehicle in ovehicles])):
        
        T = params.T
        fig, axes = plt.subplots(T // 2 + (T % 2), 2, figsize=(10, (10 / 4)*T))
        axes = axes.ravel()

        """Get scene bitmap"""
        scene = pred_result.scene
        map_mask = scene.map['VISUALIZATION'].as_image()
        map_data = util.AttrDict()
        map_data.road_bitmap = np.max(map_mask, axis=2)
        map_data.road_div_bitmap = map_mask[..., 1]
        map_data.lane_div_bitmap = map_mask[..., 0]
        map_data.extent = (scene.x_min, scene.x_max, scene.y_min, scene.y_max)

        """Get control trajectory data"""
        X = np.concatenate((ctrl_result.start[None], ctrl_result.X_star[traj_idx, :, :2]), axis=0)
        headings = []
        for t in range(1, T + 1):
            heading = np.arctan2(X[t, 1] - X[t - 1, 1], X[t, 0] - X[t - 1, 0])
            headings.append(heading)
        headings = np.array(headings)

        for t, ax in zip(range(T), axes):
            _plot_multiple_coinciding_controls_timestep(ax, map_data, ovehicles,
                    params, ctrl_result, X, headings, t, traj_idx, latent_indices, ego_bbox)
        fig.tight_layout()
        fig.savefig(os.path.join('out', f"{filename}_traj{traj_idx + 1}.png"))
        fig.clf()

def plot_multiple_coinciding_controls_timestep(ax, scene, ovehicles,
        params, ctrl_result, X_star, headings, t, traj_idx, latent_indices,
        ego_bbox, extent=None):
    ovehicle_colors = get_ovehicle_color_set()
    render_scene(ax, scene, global_coordinates=True)
    ax.plot(ctrl_result.start[0], ctrl_result.start[1],
            marker='*', markersize=8, color="blue")
    ax.plot(ctrl_result.goal[0], ctrl_result.goal[1],
            marker='*', markersize=8, color="green")

    # Plot ego vehicle trajectory
    ax.plot(X_star[:(t + 2), 0], X_star[:(t + 2), 1], 'k-o', markersize=2)

    # Get vertices of EV and plot its bounding box
    vertices = get_vertices_from_center(
            ctrl_result.X_star[traj_idx, t, :2], headings[t], ego_bbox)
    bb = patches.Polygon(vertices.reshape((-1,2,)), closed=True, color='k', fc='none')
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

def plot_multiple_coinciding_controls(pred_result, ovehicles,
        params, ctrl_result, ego_bbox, filename='lcss_control'):
    """
    Parameters
    ==========
    predict_result : util.AttrDict
        Payload containing scene, timestep, nodes, predictions, z,
        latent_probs, past_dict, ground_truth_dict
    ovehicles : list of OVehicle
    params : util.AttrDict
    ctrl_result : util.AttrDict
    ego_bbox : list of int
    filename : str
    """

    """Plots for paper"""
    for traj_idx, latent_indices in enumerate(
            util.product_list_of_list([range(ovehicle.n_states) for ovehicle in ovehicles])):
        
        T = params.T
        fig, axes = plt.subplots(T // 2 + (T % 2), 2, figsize=(10, (10 / 4)*T))
        axes = axes.ravel()

        """Get control trajectory data"""
        X_star = np.concatenate((ctrl_result.start[None], ctrl_result.X_star[traj_idx, :, :2]), axis=0)
        x_min, y_min = np.min(X_star, axis=0) - 20
        x_max, y_max = np.max(X_star, axis=0) + 20
        extent = (x_min, x_max, y_min, y_max)
        headings = []
        for t in range(1, T + 1):
            heading = np.arctan2(X_star[t, 1] - X_star[t - 1, 1], X_star[t, 0] - X_star[t - 1, 0])
            headings.append(heading)
        headings = np.array(headings)

        for t, ax in zip(range(T), axes):
            plot_multiple_coinciding_controls_timestep(ax, pred_result.scene, ovehicles,
                    params, ctrl_result, X_star, headings, t, traj_idx, latent_indices,
                    ego_bbox, extent=extent)
        fig.tight_layout()
        fig.savefig(os.path.join('out', f"{filename}_traj{traj_idx + 1}.png"))
        fig.clf()

def plot_multiple_simulation(scene, actual_trajectory, planned_trajectories,
        planned_controls, road_segs, ego_bbox, control_horizon,
        filename="oa_simulation"):
    """Plot to compare between actual trajectory and planned trajectories"""

    scene_df = scene_to_df(scene)
    scene_df[['position_x', 'position_y']] += np.array([scene.x_min, scene.y_min])
    node_ids = scene_df['node_id'].unique()
    frames, actual_trajectory = util.unzip([i for i in actual_trajectory.items()])
    frames = np.array(frames)
    actual_trajectory = np.stack(actual_trajectory)
    actual_xy = actual_trajectory[:, :2]
    planned_frames, planned_trajectories = util.unzip([i for i in planned_trajectories.items()])
    
    N = len(planned_trajectories)
    fig, axes = plt.subplots(N // 2 + (N % 2), 2, figsize=(10, (10 / 4)*N))
    axes = axes.ravel()

    for ax in axes:
        """Render road overlay and plot actual trajectory"""
        render_scene(ax, scene, global_coordinates=True)
        for A, b in road_segs.polytopes:
            util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.2)
    
    for idx, node_id in enumerate(node_ids):
        """Plot OV trajectory"""
        if node_id == "ego": continue
        node_df = scene_df[scene_df['node_id'] == node_id]
        X = node_df[['position_x', 'position_y']].values.T
        for ax in axes:
            ax.plot(X[0], X[1], ':.', color=AGENT_COLORS[idx % NCOLORS])
    
    for ax, planned_frame, planned_trajectory in zip(axes, planned_frames, planned_trajectories):
        """Plot planned trajectories on separate plots"""
        idx = np.argwhere(frames == planned_frame)[0, 0] + 1
        # plans are made in current frame, and carried out next frame
        ax.plot(actual_xy[:idx, 0], actual_xy[:idx, 1], '-ko')
        ax.plot(actual_xy[idx:(idx + control_horizon), 0], actual_xy[idx:(idx + control_horizon), 1], '-o', color="orange")
        ax.plot(actual_xy[(idx + control_horizon):, 0], actual_xy[(idx + control_horizon):, 1], '-ko')
        # planned_trajectory has shape (N_select, T, nx)
        for _planned_trajectory in planned_trajectory:
            planned_xy = _planned_trajectory[:, :2]
            ax.plot(planned_xy.T[0], planned_xy.T[1], "--b.", zorder=20)
        _planned_xy = planned_trajectory.reshape(-1, planned_trajectory.shape[2])
        min_x, min_y = np.min(_planned_xy[:, :2], axis=0) - 20
        max_x, max_y = np.max(_planned_xy[:, :2], axis=0) + 20
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))
    fig.clf()

#######################################################
# Methods for probabilistic multiple coinciding control
#######################################################

