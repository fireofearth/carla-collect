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
from ...visualize.trajectron import render_scene, render_map_crop
from ...trajectron import scene_to_df

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

class PlotPredictiveControl(object):
    """ """

    def __init__(
        self, pred_result, ovehicles, params, ctrl_result,
        T, ego_bbox, T_coin=None, grid_shape="wide"
    ):
        """Plot predictions and control trajectory
        for contingency planning predictive control.

        Parameters
        ==========
        predict_result : util.AttrDict
            Prediction payload containing scene, timestep, nodes,
            predictions, z, latent_probs, past_dict, ground_truth_dict.
        ovehicles : list of OVehicle
            Data representing other vehicles.
        params : util.AttrDict
            Parameters of MPC step including predictions and past.
        ctrl_result : util.AttrDict
            Predictive control payload containing cost, U_star, X_star,
            goal, A_union, b_union, vertices, segments.
        T : int
            Control horizon.
        ego_bbox : list of int
            EV bounding box.
        T_coin : int
            Coinciding horizon if applicable.
        grid_shape : str
            Shape of plot. Can be 'wide' or 'tall'.
        """
        self.pred_result = pred_result
        self.ovehicles = ovehicles
        self.params = params
        self.ctrl_result = ctrl_result
        self.T = T
        self.ego_bbox = ego_bbox
        self.T_coin=T_coin
        self.grid_shape = grid_shape
    
    def __render_scene_bev(self, ax):
        render_scene(ax, self.pred_result.scene, global_coordinates=True)
    
    def __plot_goal(self, ax):
        ax.plot(self.ctrl_result.goal[0], self.ctrl_result.goal[1],
                marker='*', markersize=8, color="yellow")

    def __plot_safe_region(self, ax):
        polytopes = util.compress(
            self.ctrl_result.segments.polytopes,
            ~self.ctrl_result.segments.mask
        )
        for A, b in polytopes:
            util.npu.plot_h_polyhedron(ax, A, b, fc='none', ec='b')

    def __plot_ev_past_trajectory(self, ax):
        """Plot ego vehicle past trajectory."""
        past = None
        minpos = np.array([self.pred_result.scene.x_min, self.pred_result.scene.y_min])
        for node in self.pred_result.nodes:
            if node.id == 'ego':
                past = self.pred_result.past_dict[self.pred_result.timestep][node] + minpos
                break
        ax.plot(past[:, 0], past[:, 1], '-ko', markersize=2)

    def __plot_ev_current_bbox(self, ax):
        """Box of current ego vehicle position."""
        vertices = util.vertices_from_bbox(
            self.params.initial_state.world[:2],
            self.params.initial_state.world[2],
            self.ego_bbox
        )
        bb = patches.Polygon(
            vertices.reshape((-1,2,)), closed=True, color='k',
            fc=util.plu.modify_alpha("black", 0.2), ls='-'
        )
        ax.add_patch(bb)

    def __plot_ev_planned_trajectory(self, ax, t, X_star):
        # Plot ego vehicle planned trajectory
        ax.plot(X_star[:(t + 2), 0], X_star[:(t + 2), 1], 'k-o', markersize=2)
        if self.T_coin is not None and t >= self.T_coin:
            ax.scatter(X_star[t + 1, 0], X_star[t + 1, 1], color='k', s=7)
    
    def __plot_ev_predicted_bbox(self, ax, t, X_star):
        """Get vertices of EV and plot its bounding box."""
        if t < 2:
            vertices = util.vertices_from_bbox(
                X_star[t + 1, :2], X_star[t + 1, 2], self.ego_bbox
            )
        else:
            """Make EV bbox look better on far horizons by fixing the heading angle."""
            heading = np.arctan2(
                X_star[t + 1, 1] - X_star[t, 1], X_star[t + 1, 0] - X_star[t, 0]
            )
            vertices = util.vertices_from_bbox(
                X_star[t + 1, :2], heading, self.ego_bbox
            )
        bb = patches.Polygon(
            vertices.reshape((-1,2,)), closed=True, color='k', fc='none', ls='-'
        )
        ax.add_patch(bb)

    def __plot_oa_ov_clusters(self, ax, t):
        """Plot other vehicles and their predictions when using OA motion planner"""
        ovehicle_colors = get_ovehicle_color_set()
        for ov_idx, ovehicle in enumerate(self.ovehicles):
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
                A = self.ctrl_result.A_union[t][latent_idx][ov_idx]
                b = self.ctrl_result.b_union[t][latent_idx][ov_idx]
                util.npu.plot_h_polyhedron(ax, A, b, ec=color, ls='-', alpha=1)

                # Plot vertices
                vertices = self.ctrl_result.vertices[t][latent_idx][ov_idx]
                X = vertices[:,0:2].T
                ax.scatter(X[0], X[1], color=color, s=2)
                X = vertices[:,2:4].T
                ax.scatter(X[0], X[1], color=color, s=2)
                X = vertices[:,4:6].T
                ax.scatter(X[0], X[1], color=color, s=2)
                X = vertices[:,6:8].T
                ax.scatter(X[0], X[1], color=color, s=2)

    def __plot_mcc_ov_clusters(self, ax, traj_idx, t, latent_indices):
        """Plot other vehicles and their predictions when using MCC motion planner
        TODO: plot current bounding boxes?"""
        ovehicle_colors = get_ovehicle_color_set()
        for ov_idx, ovehicle in enumerate(self.ovehicles):
            # Plot past trajectory
            latent_idx = latent_indices[ov_idx]
            color = ovehicle_colors[ov_idx][0]
            ax.plot(ovehicle.past[:,0], ovehicle.past[:,1],
                    marker='o', markersize=2, color=color)

            # Plot overapproximation
            color = ovehicle_colors[ov_idx][latent_idx]
            A = self.ctrl_result.A_unions[traj_idx][t][ov_idx]
            b = self.ctrl_result.b_unions[traj_idx][t][ov_idx]
            util.npu.plot_h_polyhedron(ax, A, b, ec=color, alpha=1)
                
            # Plot vertices
            vertices = self.ctrl_result.vertices[t][latent_idx][ov_idx]
            X = vertices[:, 0:2].T
            ax.scatter(X[0], X[1], color=color, s=2)
            X = vertices[:, 2:4].T
            ax.scatter(X[0], X[1], color=color, s=2)
            X = vertices[:, 4:6].T
            ax.scatter(X[0], X[1], color=color, s=2)
            X = vertices[:, 6:8].T
            ax.scatter(X[0], X[1], color=color, s=2)
    
    def __set_extent(self, ax, t, extent=None):
        if extent is not None:
            ax.set_xlim([extent[0], extent[1]])
            ax.set_ylim([extent[2], extent[3]])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {t + 1}")
        ax.set_aspect("equal")

    def plot_oa_prediction_timestep(self, ax, t, X_star, extent=None):
        self.__render_scene_bev(ax)
        self.__plot_goal(ax)
        self.__plot_safe_region(ax)
        self.__plot_ev_past_trajectory(ax)
        self.__plot_ev_current_bbox(ax)
        self.__plot_ev_planned_trajectory(ax, t, X_star)
        self.__plot_ev_predicted_bbox(ax, t, X_star)
        self.__plot_oa_ov_clusters(ax, t)
        self.__set_extent(ax, t, extent=extent)

    def plot_oa_prediction(self, filename="lcss_control"):
        # X_star: ndarray
        #   Predicted controls including initial state.
        X_star = np.concatenate(
            (self.params.initial_state.world[None], self.ctrl_result.X_star)
        )
        x_min, y_min = np.min(X_star[:, :2], axis=0)
        x_max, y_max = np.max(X_star[:, :2], axis=0)
        # make extent the same size across all timesteps,
        # with planned trajectory centered
        x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
        extent = (x_mid - PADDING, x_mid + PADDING, y_mid - PADDING, y_mid + PADDING)

        """Plots for paper."""
        if self.grid_shape == "tall":
            # make the plotting grid tall
            fig, axes = plt.subplots(
                self.T // 2 + (self.T % 2), 2, figsize=(10, (10 / 4)*self.T)
            )
        elif self.grid_shape == "wide":
            # make the plotting grid wide
            fig, axes = plt.subplots(
                2, self.T // 2 + (self.T % 2), figsize=((10 / 4)*self.T, 10)
            )
        else:
            NotImplementedError(f"Unknown grid shape {self.grid_shape}")
        axes = axes.ravel()
        for t, ax in zip(range(self.T), axes):
            self.plot_oa_prediction_timestep(ax, t, X_star, extent=extent)
        fig.tight_layout()
        fig.savefig(os.path.join('out', f"{filename}.png"))
        fig.clf()
    
    def plot_mcc_prediction_timestep(
        self, ax, traj_idx, t, latent_indices, X_star, extent=None
    ):
        """Helper function for plot_multiple_coinciding_controls()
    
        Parameters
        ==========
        ax : matplotlib.axes.Axes
            The plot to make.
        traj_idx : int
            Index of trajectory to plot.
        t : int
            The timestep of predicted state we want to show for this plot.
            The state corresponding to timestep 1 is X_star[t + 1].
        latent_indices : ndarray of int
            Indices of latent corresponding to each vehicle on the scene.
        X_star : ndarray
            The predicted states including initial state from optimization.
        extent : tuple of number
            Extent of plot (x min, x max, y min, y max).
        """
        self.__render_scene_bev(ax)
        self.__plot_goal(ax)
        self.__plot_safe_region(ax)
        self.__plot_ev_past_trajectory(ax)
        self.__plot_ev_current_bbox(ax)
        self.__plot_ev_planned_trajectory(ax, t, X_star)
        self.__plot_ev_predicted_bbox(ax, t, X_star)
        self.__plot_mcc_ov_clusters(ax, traj_idx, t, latent_indices)
        self.__set_extent(ax, t, extent=extent)
    
    def plot_mcc_prediction(self, filename="mcc_control"):
        # X_star: ndarray
        #   Predicted controls including initial state.
        X_init = np.repeat(
            self.params.initial_state.world[None], self.params.N_select, axis=0
        )
        X_star = np.concatenate((X_init[:, None], self.ctrl_result.X_star), axis=1)
        _X_star = X_star[..., :2].reshape(-1, 2)
        x_min, y_min = np.min(_X_star, axis=0)
        x_max, y_max = np.max(_X_star, axis=0)
        x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
        extent = (x_mid - PADDING, x_mid + PADDING, y_mid - PADDING, y_mid + PADDING)
        for traj_idx in range(self.params.N_select):
            """Generate a single plot for each combination of overapproximations
            that we have applied control over."""
            if self.grid_shape == "tall":
                # make the plotting grid tall
                fig, axes = plt.subplots(
                    self.T // 2 + (self.T % 2), 2, figsize=(10, (10 / 4)*self.T)
                )
            elif self.grid_shape == "wide":
                # make the plotting grid wide
                fig, axes = plt.subplots(
                    2, self.T // 2 + (self.T % 2), figsize=((10 / 4)*self.T, 10)
                )
            else:
                NotImplementedError(f"Unknown grid shape {self.grid_shape}")
            axes = axes.ravel()
            latent_indices = self.params.sublist_joint_decisions[traj_idx]
            for t, ax in enumerate(axes):
                self.plot_mcc_prediction_timestep(
                    ax, traj_idx, t, latent_indices, X_star, extent=extent
                )
            fig.tight_layout()
            fig.savefig(os.path.join('out', f"{filename}_traj{traj_idx + 1}.png"))
            fig.clf()
    
    def plot_failure_timestep(self, ax, t, extent=None):
        self.__render_scene_bev(ax)
        self.__plot_goal(ax)
        self.__plot_safe_region(ax)
        self.__plot_ev_past_trajectory(ax)
        self.__plot_ev_current_bbox(ax)
        self.__plot_oa_ov_clusters(ax, t)
        self.__set_extent(ax, t, extent=extent)

    def plot_oa_failure(self, filename="optim_fail"):
        x_min, y_min = self.params.initial_state.world[:2] - 2*PADDING
        x_max, y_max = self.params.initial_state.world[:2] + 2*PADDING
        extent = (x_min, x_max, y_min, y_max)

        """Plots for paper."""
        if self.grid_shape == "tall":
            # make the plotting grid tall
            fig, axes = plt.subplots(
                self.T // 2 + (self.T % 2), 2, figsize=(10, (10 / 4)*self.T)
            )
        elif self.grid_shape == "wide":
            # make the plotting grid wide
            fig, axes = plt.subplots(
                2, self.T // 2 + (self.T % 2), figsize=((10 / 4)*self.T, 10)
            )
        else:
            NotImplementedError(f"Unknown grid shape {self.grid_shape}")
        axes = axes.ravel()
        for t, ax in zip(range(self.T), axes):
            self.plot_prediction_timestep(ax, t, extent=extent)
        fig.tight_layout()
        fig.savefig(os.path.join("out", f"{filename}.png"))
        fig.clf()

    def plot_mcc_failure_timestep(
        self, ax, traj_idx, t, latent_indices, extent=None
    ):
        self.__render_scene_bev(ax)
        self.__plot_goal(ax)
        self.__plot_safe_region(ax)
        self.__plot_ev_past_trajectory(ax)
        self.__plot_ev_current_bbox(ax)
        self.__plot_mcc_ov_clusters(ax, traj_idx, t, latent_indices)
        self.__set_extent(ax, t, extent=extent)

    def plot_mcc_prediction(self, filename="mcc_optim_fail"):
        x_min, y_min = self.params.initial_state.world[:2] - 2*PADDING
        x_max, y_max = self.params.initial_state.world[:2] + 2*PADDING
        extent = (x_min, x_max, y_min, y_max)
        for traj_idx in range(self.params.N_select):
            """Generate a single plot for each combination of overapproximations
            that we have applied control over."""
            if self.grid_shape == "tall":
                # make the plotting grid tall
                fig, axes = plt.subplots(
                    self.T // 2 + (self.T % 2), 2, figsize=(10, (10 / 4)*self.T)
                )
            elif self.grid_shape == "wide":
                # make the plotting grid wide
                fig, axes = plt.subplots(
                    2, self.T // 2 + (self.T % 2), figsize=((10 / 4)*self.T, 10)
                )
            else:
                NotImplementedError(f"Unknown grid shape {self.grid_shape}")
            axes = axes.ravel()
            latent_indices = self.params.sublist_joint_decisions[traj_idx]
            for t, ax in enumerate(axes):
                self.plot_mcc_failure_timestep(
                    ax, traj_idx, t, latent_indices, extent=extent
                )
            fig.tight_layout()
            fig.savefig(os.path.join('out', f"{filename}_traj{traj_idx + 1}.png"))
            fig.clf()

class PlotOASimulation(object):
    def __init__(
        self, scene, map_data, actual_trajectory, planned_trajectories, planned_controls,
        goals, lowlevel, segments, ego_bbox, step_horizon, steptime
    ):
        """Constructor.
        
        Parameters
        ==========
        map_data : util.AttrDict
            Container of vertices for road segments and lines produced by MapQuerier.
        actual_trajectory : collections.OrderedDict of (int, ndarray)
            Indexed by frame ID. Array of EV global (x, y) position, heading and speed.
        planned_trajectories : collections.OrderedDict of (int, ndarray)
            Indexed by frame ID. The EV's planned state over timesteps T+1 incl. origin
            with global (x, y) position, heading and speed as ndarray of shape (T + 1, 4).
        planned_controls : collections.OrderedDict of (int, ndarray)
            Indexed by frame ID. The EV's planned controls over timesteps T with
            acceleration, steering as ndarray of shape (T, 2).
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
        
        TODO: fix incompatibility between segments, road_segs
        """
        self.scene = scene
        self.map_data = map_data
        self.actual_trajectory = actual_trajectory
        self.planned_trajectories = planned_trajectories
        self.planned_controls = planned_controls
        self.goals = goals
        self.lowlevel = lowlevel
        self.segments = segments
        self.ego_bbox = ego_bbox
        self.step_horizon = step_horizon, steptime
        pass