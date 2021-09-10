import os
import logging

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
import carla

import utility as util
import carlautil
import carlautil.debug

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
    """Gets A_t, b_0 for the contraint set A_t x >= b_0
    vertices : np.array
        Vertices of shape (?, 8)
    
    Returns
    =======
    np.array
        A_t matrix of shape (4, 2)
    np.array
        b_0 vector of shape (4,)
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

def get_ovehicle_color_set():
    OVEHICLE_COLORS = [
        clr.LinearSegmentedColormap.from_list('ro', ['red', 'orange'], N=256),
        clr.LinearSegmentedColormap.from_list('gy', ['green', 'yellow'], N=256),
        clr.LinearSegmentedColormap.from_list('bp', ['blue', 'purple'], N=256),
        clr.LinearSegmentedColormap.from_list('td', ['turquoise', 'deeppink'], N=256),
        clr.LinearSegmentedColormap.from_list('bt', ['brown', 'teal'], N=256),
    ]
    ovehicle_colors = []
    for ov_colormap in OVEHICLE_COLORS:
        ov_colors = ov_colormap(np.linspace(0,1,5))
        ovehicle_colors.append(ov_colors)
    return ovehicle_colors

def plot_lcss_prediction_timestep(ax, map_data, ovehicles,
        params, ctrl_result, t, ego_bbox):
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

    # Plot ego vehicle
    ax.plot(ctrl_result.X_star[:t, 0], ctrl_result.X_star[:t, 1],
            'k-o', markersize=2)

    # Get vertices of EV and plot its bounding box
    vertices = get_vertices_from_center(
            ctrl_result.X_star[t],
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
    fig, axes = plt.subplots(T // 2 + (T % 2), 2, figsize=(10, 20))
    axes = axes.ravel()

    """Get scene bitmap"""
    scene = pred_result.scene
    map_mask = scene.map['VISUALIZATION'].as_image()
    map_data = util.AttrDict()
    map_data.road_bitmap = np.max(map_mask, axis=2)
    map_data.road_div_bitmap = map_mask[..., 1]
    map_data.lane_div_bitmap = map_mask[..., 0]
    map_data.extent = (scene.x_min, scene.x_max, scene.y_min, scene.y_max)

    for t, ax in zip(range(T), axes):
        plot_lcss_prediction_timestep(ax, map_data, ovehicles,
                params, ctrl_result, t, ego_bbox)
    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))

#########################################
# Methods for multiple coinciding control
#########################################

def plot_multiple_coinciding_controls_timestep(ax, map_data, ovehicles,
        params, ctrl_result, X, headings, t, ego_bbox):
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
    N_traj = params.N_traj

    for idx in range(N_traj):
        # Plot ego vehicle trajectory
        ax.plot(ctrl_result.X_star[idx, :t, 0],
                ctrl_result.X_star[idx, :t, 1], 'k-o', markersize=2)
        # Get vertices of EV and plot its bounding box
        vertices = get_vertices_from_center(
                ctrl_result.X_star[idx, t, :2], headings[idx, t], ego_bbox)
        bb = patches.Polygon(vertices.reshape((-1,2,)),
                closed=True, color='k', fc='none')
        ax.add_patch(bb)

    # Plot other vehicles
    for ov_idx, ovehicle in enumerate(ovehicles):
        color = ovehicle_colors[ov_idx][0]
        ax.plot(ovehicle.past[:,0], ovehicle.past[:,1],
                marker='o', markersize=2, color=color)
        for traj_idx in range(params.N_traj):
            color = ovehicle_colors[ov_idx][0]

            # Plot overapproximation
            A = ctrl_result.A_unions[traj_idx][t][ov_idx]
            b = ctrl_result.b_unions[traj_idx][t][ov_idx]
            try:
                plot_h_polyhedron(ax, A, b, ec=color, alpha=1)
            except scipy.spatial.qhull.QhullError as e:
                print(f"Failed to plot polyhedron at timestep t={t}")
        
        for latent_idx in range(ovehicle.n_states):
            color = ovehicle_colors[ov_idx][latent_idx]
            
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
    T = params.T
    fig, axes = plt.subplots(T // 2 + (T % 2), 2, figsize=(10, 20))
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
    N_traj, T = params.N_traj, params.T
    start = np.tile(ctrl_result.start, (N_traj, 1, 1))
    X = np.concatenate((start, ctrl_result.X_star[..., :2]), axis=1)
    headings = []
    for t in range(1, T + 1):
        heading = np.arctan2(X[:, t, 1] - X[:, t - 1, 1], X[:, t, 0] - X[:, t - 1, 0])
        headings.append(heading)
    # headings has shape (N_traj, T)
    headings = np.stack(headings, axis=1)

    for t, ax in zip(range(T), axes):
        plot_multiple_coinciding_controls_timestep(ax, map_data, ovehicles,
                params, ctrl_result, X, headings, t, ego_bbox)
    fig.tight_layout()
    fig.savefig(os.path.join('out', f"{filename}.png"))

