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
import mpl_toolkits
import mpl_toolkits.axes_grid1
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
import utility.npu as npu
import utility.plu
import carlautil
import carlautil.debug


AGENT_COLORS = [
    "blue",
    "darkviolet",
    "dodgerblue",
    "darkturquoise",
    "green",
    "gold",
    "orange",
    "red",
    "deeppink",
]
AGENT_COLORS = np.array(AGENT_COLORS).take(
    [(i * 5) % len(AGENT_COLORS) for i in range(17)], 0
)
NCOLORS = len(AGENT_COLORS)
PADDING = 20

OVEHICLE_COLORS = [
    clr.LinearSegmentedColormap.from_list("ro", ["red", "orange"], N=256),
    clr.LinearSegmentedColormap.from_list("gy", ["green", "yellow"], N=256),
    clr.LinearSegmentedColormap.from_list("bp", ["blue", "purple"], N=256),
    clr.LinearSegmentedColormap.from_list("td", ["turquoise", "deeppink"], N=256),
    clr.LinearSegmentedColormap.from_list("bt", ["brown", "teal"], N=256),
]


def get_ovehicle_color_set(latents=None):
    latents = [] if latents is None else latents
    ovehicle_colors = []
    for idx, ov_colormap in enumerate(OVEHICLE_COLORS):
        try:
            l = latents[idx]
        except IndexError:
            l = 5
        ov_colors = ov_colormap(np.linspace(0, 1, l))
        ovehicle_colors.append(ov_colors)
    return ovehicle_colors


class PlotPredictiveControl(object):
    """Plot predictions and control trajectory
    for contingency planning predictive control."""

    PADDING = 40

    def __init__(
        self,
        pred_result,
        ovehicles,
        params,
        ctrl_result,
        T,
        ego_bbox,
        T_coin=None,
        grid_shape="wide",
    ):
        """Constructor.

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
        self.T_coin = T_coin
        self.grid_shape = grid_shape

    def __render_scene_bev(self, ax):
        render_scene(ax, self.pred_result.scene, global_coordinates=True)

    def __plot_goal(self, ax):
        ax.plot(
            self.ctrl_result.goal[0],
            self.ctrl_result.goal[1],
            marker="*",
            markersize=8,
            color="yellow",
        )

    def __plot_safe_region(self, ax):
        """Plot safe region if it exists, otherwise skip."""
        if "segments" not in self.ctrl_result:
            return
        polytopes = util.compress(
            self.ctrl_result.segments.polytopes, ~self.ctrl_result.segments.mask
        )
        for A, b in polytopes:
            util.npu.plot_h_polyhedron(ax, A, b, fc="none", ec="b")

    def __plot_ev_past_trajectory(self, ax):
        """Plot ego vehicle past trajectory."""
        past = None
        minpos = np.array([self.pred_result.scene.x_min, self.pred_result.scene.y_min])
        for node in self.pred_result.nodes:
            if node.id == "ego":
                past = (
                    self.pred_result.past_dict[self.pred_result.timestep][node] + minpos
                )
                break
        ax.plot(past[:, 0], past[:, 1], "-ko", markersize=2)

    def __plot_ev_current_bbox(self, ax):
        """Box of current ego vehicle position."""
        vertices = util.vertices_from_bbox(
            self.params.initial_state.world[:2],
            self.params.initial_state.world[2],
            self.ego_bbox,
        )
        bb = patches.Polygon(
            vertices.reshape(
                (
                    -1,
                    2,
                )
            ),
            closed=True,
            color="k",
            fc=util.plu.modify_alpha("black", 0.2),
            ls="-",
        )
        ax.add_patch(bb)

    def __plot_ev_planned_trajectory(self, ax, t, X_star):
        # Plot ego vehicle planned trajectory
        ax.plot(X_star[: (t + 2), 0], X_star[: (t + 2), 1], "k-o", markersize=2)
        if self.T_coin is not None and t >= self.T_coin:
            ax.scatter(X_star[t + 1, 0], X_star[t + 1, 1], color="k", s=7)

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
            vertices.reshape(
                (
                    -1,
                    2,
                )
            ),
            closed=True,
            color="k",
            fc="none",
            ls="-",
        )
        ax.add_patch(bb)

    def __plot_oa_ov_clusters(self, ax, t):
        """Plot other vehicles and their predictions when using OA motion planner"""
        ovehicle_colors = get_ovehicle_color_set()
        for ov_idx, ovehicle in enumerate(self.ovehicles):
            color = ovehicle_colors[ov_idx][0]
            ax.plot(
                ovehicle.past[:, 0],
                ovehicle.past[:, 1],
                marker="o",
                markersize=2,
                color=color,
            )
            heading = np.arctan2(
                ovehicle.past[-1, 1] - ovehicle.past[-2, 1],
                ovehicle.past[-1, 0] - ovehicle.past[-2, 0],
            )
            vertices = util.vertices_from_bbox(
                ovehicle.past[-1], heading, np.array([ovehicle.bbox])
            )
            bb = patches.Polygon(
                vertices.reshape(
                    (
                        -1,
                        2,
                    )
                ),
                closed=True,
                color=color,
                fc=util.plu.modify_alpha(color, 0.2),
                ls="-",
            )
            ax.add_patch(bb)
            for latent_idx in range(ovehicle.n_states):
                color = ovehicle_colors[ov_idx][latent_idx]

                # Plot overapproximation
                A = self.ctrl_result.A_union[t][latent_idx][ov_idx]
                b = self.ctrl_result.b_union[t][latent_idx][ov_idx]
                util.npu.plot_h_polyhedron(ax, A, b, ec=color, ls="-", alpha=1)

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

    def __plot_mcc_ov_clusters(self, ax, traj_idx, t, latent_indices):
        """Plot other vehicles and their predictions when using MCC motion planner
        TODO: plot current bounding boxes?"""
        ovehicle_colors = get_ovehicle_color_set()
        for ov_idx, ovehicle in enumerate(self.ovehicles):
            # Plot past trajectory
            latent_idx = latent_indices[ov_idx]
            color = ovehicle_colors[ov_idx][0]
            ax.plot(
                ovehicle.past[:, 0],
                ovehicle.past[:, 1],
                marker="o",
                markersize=2,
                color=color,
            )

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
        extent = (
            x_mid - self.PADDING,
            x_mid + self.PADDING,
            y_mid - self.PADDING,
            y_mid + self.PADDING
        )
        """Plots for paper."""
        if self.grid_shape == "tall":
            # make the plotting grid tall
            fig, axes = plt.subplots(
                self.T // 2 + (self.T % 2), 2, figsize=(10, (10 / 4) * self.T)
            )
        elif self.grid_shape == "wide":
            # make the plotting grid wide
            fig, axes = plt.subplots(
                2, self.T // 2 + (self.T % 2), figsize=((10 / 4) * self.T, 10)
            )
        else:
            NotImplementedError(f"Unknown grid shape {self.grid_shape}")
        axes = axes.ravel()
        for t, ax in zip(range(self.T), axes):
            self.plot_oa_prediction_timestep(ax, t, X_star, extent=extent)
        fig.tight_layout()
        fig.savefig(os.path.join("out", f"{filename}.png"))
        plt.close(fig)

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
        extent = (
            x_mid - self.PADDING,
            x_mid + self.PADDING,
            y_mid - self.PADDING,
            y_mid + self.PADDING
        )
        for traj_idx in range(self.params.N_select):
            """Generate a single plot for each combination of overapproximations
            that we have applied control over."""
            if self.grid_shape == "tall":
                # make the plotting grid tall
                fig, axes = plt.subplots(
                    self.T // 2 + (self.T % 2), 2, figsize=(10, (10 / 4) * self.T)
                )
            elif self.grid_shape == "wide":
                # make the plotting grid wide
                fig, axes = plt.subplots(
                    2, self.T // 2 + (self.T % 2), figsize=((10 / 4) * self.T, 10)
                )
            else:
                NotImplementedError(f"Unknown grid shape {self.grid_shape}")
            axes = axes.ravel()
            latent_indices = self.params.sublist_joint_decisions[traj_idx]
            for t, ax in enumerate(axes):
                self.plot_mcc_prediction_timestep(
                    ax, traj_idx, t, latent_indices, X_star[traj_idx], extent=extent
                )
            fig.tight_layout()
            fig.savefig(os.path.join("out", f"{filename}_traj{traj_idx + 1}.png"))
            plt.close(fig)

    def plot_oa_failure_timestep(self, ax, t, extent=None):
        self.__render_scene_bev(ax)
        self.__plot_goal(ax)
        self.__plot_safe_region(ax)
        self.__plot_ev_past_trajectory(ax)
        self.__plot_ev_current_bbox(ax)
        self.__plot_oa_ov_clusters(ax, t)
        self.__set_extent(ax, t, extent=extent)

    def plot_oa_failure(self, filename="oa_optim_fail"):
        x_min, y_min = self.params.initial_state.world[:2] - self.PADDING
        x_max, y_max = self.params.initial_state.world[:2] + self.PADDING
        extent = (x_min, x_max, y_min, y_max)

        """Plots for paper."""
        if self.grid_shape == "tall":
            # make the plotting grid tall
            fig, axes = plt.subplots(
                self.T // 2 + (self.T % 2), 2, figsize=(10, (10 / 4) * self.T)
            )
        elif self.grid_shape == "wide":
            # make the plotting grid wide
            fig, axes = plt.subplots(
                2, self.T // 2 + (self.T % 2), figsize=((10 / 4) * self.T, 10)
            )
        else:
            NotImplementedError(f"Unknown grid shape {self.grid_shape}")
        axes = axes.ravel()
        for t, ax in zip(range(self.T), axes):
            self.plot_oa_failure_timestep(ax, t, extent=extent)
        fig.tight_layout()
        fig.savefig(os.path.join("out", f"{filename}.png"))
        plt.close(fig)

    def plot_mcc_failure_timestep(self, ax, traj_idx, t, latent_indices, extent=None):
        self.__render_scene_bev(ax)
        self.__plot_goal(ax)
        self.__plot_safe_region(ax)
        self.__plot_ev_past_trajectory(ax)
        self.__plot_ev_current_bbox(ax)
        self.__plot_mcc_ov_clusters(ax, traj_idx, t, latent_indices)
        self.__set_extent(ax, t, extent=extent)

    def plot_mcc_failure(self, filename="mcc_optim_fail"):
        x_min, y_min = self.params.initial_state.world[:2] - self.PADDING
        x_max, y_max = self.params.initial_state.world[:2] + self.PADDING
        extent = (x_min, x_max, y_min, y_max)
        for traj_idx in range(self.params.N_select):
            """Generate a single plot for each combination of overapproximations
            that we have applied control over."""
            if self.grid_shape == "tall":
                # make the plotting grid tall
                fig, axes = plt.subplots(
                    self.T // 2 + (self.T % 2), 2, figsize=(10, (10 / 4) * self.T)
                )
            elif self.grid_shape == "wide":
                # make the plotting grid wide
                fig, axes = plt.subplots(
                    2, self.T // 2 + (self.T % 2), figsize=((10 / 4) * self.T, 10)
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
            fig.savefig(os.path.join("out", f"{filename}_traj{traj_idx + 1}.png"))
            plt.close(fig)


class PlotSimulation(object):
    
    def __init__(
        self,
        scene,
        map_data,
        actual_trajectory,
        planned_trajectories,
        planned_controls,
        goals,
        lowlevel,
        segments,
        ego_bbox,
        T_step,
        steptime,
        T_coin=None,
        filename="oa_simulation",
        road_boundary_constraints=True,
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
        T_step : int
            Number of predictions steps to execute at each iteration of MPC.
        steptime : float
            Time in seconds taken to complete one step of MPC.
        T_coin : int
            Coinciding horizon if applicable.
        filename : str
            Partial file name to save plots.
        road_boundary_constraints : bool
            Whether to visualize boundary constrains in plots.
        """
        self.scene = scene
        self.map_data = map_data
        self.lowlevel = lowlevel
        self.segments = segments
        self.ego_bbox = ego_bbox
        self.steptime = steptime
        self.filename = filename
        self.T_step = T_step
        self.T_coin = T_coin
        self.road_boundary_constraints = road_boundary_constraints
        self.scene_df = scene_to_df(self.scene)
        self.scene_df[["position_x", "position_y"]] += np.array(
            [self.scene.x_min, self.scene.y_min]
        )
        self.node_ids = self.scene_df["node_id"].unique()
        frames, actual_trajectory = util.unzip([i for i in actual_trajectory.items()])
        self.frames = np.array(frames)
        self.actual_trajectory = np.stack(actual_trajectory)
        self.planned_frames, self.planned_controls = util.unzip(
            [i for i in planned_controls.items()]
        )
        _, self.planned_trajectories = util.unzip(
            [i for i in planned_trajectories.items()]
        )
        _, self.goals = util.unzip([i for i in goals.items()])

    def __render_scene_bev(self, ax, goal, extent):
        """Map overlay"""
        render_map_crop(ax, self.map_data, extent)
        if self.road_boundary_constraints:
            if "mask" in self.segments:
                for (A, b), in_junction in zip(
                    self.segments.polytopes, self.segments.mask
                ):
                    if in_junction:
                        util.npu.plot_h_polyhedron(ax, A, b, fc="r", ec="r", alpha=0.2)
                    else:
                        util.npu.plot_h_polyhedron(ax, A, b, fc="b", ec="b", alpha=0.2)
            else:
                for A, b in self.segments.polytopes:
                    util.npu.plot_h_polyhedron(ax, A, b, fc="b", ec="b", alpha=0.2)
        ax.plot(goal.x, goal.y, marker="*", markersize=8, color="yellow")

    def __plot_ov_trajectory(self, ax):
        for idx, node_id in enumerate(self.node_ids):
            """Plot OV's trajectory"""
            if node_id == "ego":
                continue
            node_df = self.scene_df[self.scene_df["node_id"] == node_id]
            X = node_df[["position_x", "position_y"]].values.T
            ax.plot(X[0], X[1], ":.", color=AGENT_COLORS[idx % NCOLORS])

    def __plot_ev_current_bbox(self, ax, frame_idx):
        "EV bounding box at current position"
        position = self.actual_trajectory[frame_idx, :2]
        heading = self.actual_trajectory[frame_idx, 2]
        vertices = util.npu.vertices_from_bbox(position, heading, self.ego_bbox)
        bb = patches.Polygon(
            vertices.reshape((-1, 2,))  # fmt: skip
            ,
            closed=True,
            color="k",
            fc=util.plu.modify_alpha("black", 0.2),
        )
        ax.add_patch(bb)

    def __plot_ev_planned_trajectory(self, ax, planned_xy):
        """Plot EV planned trajectory. This is computed from
        CPLEX using linearized discrete-time vehicle dynamics."""
        ax.plot(*planned_xy.T, "--bo", zorder=20, markersize=2)

    def __plot_ev_planned_coin_trajectory(self, ax, traj_colors, coin_xy, cont_xy):
        """Plot EV planned contingency trajectory. This is computed
        from CPLEX using linearized discrete-time vehicle dynamics."""
        n_select = cont_xy.shape[0]
        ax.plot(*coin_xy.T, "--bo", zorder=20, markersize=2)
        for traj_idx in range(n_select):
            # plot contingency plans separately
            ax.plot(
                *cont_xy[traj_idx].T,
                "--o",
                color=traj_colors[traj_idx],
                zorder=20,
                markersize=2,
            )

    def __plot_ev_actual_trajectory(self, ax, frame_idx, actual_xy):
        """Plot EV actual trajectory.
        Plans are made in current frame, and carried out next frame."""
        idx = frame_idx + 1
        ax.plot(*actual_xy[:idx].T, "-ko")
        ax.plot(*actual_xy[idx : (idx + self.T_step)].T, "-o", color="orange")
        ax.plot(*actual_xy[(idx + self.T_step) :].T, "-ko")

    def __plot_ev_planned_bbox(self, ax, planned_xy, planned_psi):
        ax.plot(*planned_xy.T, "b.", zorder=20, markersize=2)
        vertices = util.npu.vertices_of_bboxes(planned_xy, planned_psi, self.ego_bbox)
        for v in vertices:
            bb = patches.Polygon(
                v, closed=True, color="b", fc=util.plu.modify_alpha("blue", 0.2)
            )
            ax.add_patch(bb)

    def __plot_ev_planned_coin_bbox(
        self, ax, traj_colors, coin_xy, coin_psi, cont_xy, cont_psi
    ):
        n_select = cont_xy.shape[0]
        ax.plot(*coin_xy.T, "b.", zorder=20, markersize=2)
        vertices = util.npu.vertices_of_bboxes(coin_xy, coin_psi, self.ego_bbox)
        for v in vertices:
            bb = patches.Polygon(
                v, closed=True, color="b", fc=util.plu.modify_alpha("blue", 0.2)
            )
            ax.add_patch(bb)
        for traj_idx in range(n_select):
            # plot contingency plans separately
            ax.plot(
                *cont_xy[traj_idx].T,
                ".",
                zorder=20,
                markersize=2,
                color=traj_colors[traj_idx],
            )
            vertices = util.npu.vertices_of_bboxes(
                cont_xy[traj_idx], cont_psi[traj_idx], self.ego_bbox
            )
            for v in vertices:
                bb = patches.Polygon(
                    v,
                    closed=True,
                    color=traj_colors[traj_idx],
                    fc=util.plu.modify_alpha(traj_colors[traj_idx], 0.2),
                )
                ax.add_patch(bb)

    def __plot_ev_actual_bbox(self, ax, actual_xy, actual_psi):
        """Plot bounding box of EV placements across all EV steps."""
        ax.plot(*actual_xy.T, "ko")
        vertices = util.npu.vertices_of_bboxes(actual_xy, actual_psi, self.ego_bbox)
        for v in vertices:
            bb = patches.Polygon(
                v, closed=True, color="k", fc="none"
            )  # fc=util.plu.modify_alpha("black", 0.2))
            ax.add_patch(bb)

    def __set_bev(self, ax, frame_idx, extent):
        "Configure overhead plot."
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"frame {frame_idx + 1}")
        ax.set_aspect("equal")

    def __plot_v(self, ax, frame_idx, planned_v, actual_v):
        """Plot v, which is the vehicle speed."""
        idx = frame_idx + 1
        ax.plot(range(1, self.frames.size + 1), actual_v, "-k.", label="actual")
        ax.plot(range(idx, idx + planned_v.size), planned_v, "-b.", label="linear plan")
        ax.set_title("$v$ speed of c.g., m/s")
        ax.set_ylabel("m/s")

    def __plot_coin_v(self, ax, traj_colors, frame_idx, coin_v, cont_v, actual_v):
        """Plot v, which is the vehicle speed."""
        idx = frame_idx + 1
        n_select = cont_v.shape[0]
        ax.plot(range(1, self.frames.size + 1), actual_v, "-k.", label="actual")
        ax.plot(
            range(idx, idx + coin_v.size), coin_v, "-b.", label="coin. plan"
        )
        for traj_idx in range(n_select):
            ax.plot(
                range(idx + self.T_coin, idx + self.T_coin + cont_v[traj_idx].size),
                cont_v[traj_idx],
                "-.",
                color=traj_colors[traj_idx],
                label=f"cont. plan {traj_idx}",
            )
        ax.set_title("$v$ speed of c.g., m/s")
        ax.set_ylabel("m/s")

    def __plot_psi(self, ax, frame_idx, planned_psi, actual_psi):
        """Plot psi, which is the vehicle longitudinal angle in global coordinates."""
        idx = frame_idx + 1
        ax.plot(range(1, self.frames.size + 1), actual_psi, "-k.", label="actual")
        ax.plot(
            range(idx, idx + planned_psi.size), planned_psi, "-b.", label="linear plan"
        )
        ax.set_title("$\psi$ longitudinal angle, radians")
        ax.set_ylabel("rad")

    def __plot_coin_psi(
        self, ax, traj_colors, frame_idx, coin_psi, cont_psi, actual_psi
    ):
        """Plot psi, which is the vehicle longitudinal angle in global coordinates."""
        idx = frame_idx + 1
        n_select = cont_psi.shape[0]
        ax.plot(range(1, self.frames.size + 1), actual_psi, "-k.", label="actual")
        ax.plot(
            range(idx, idx + coin_psi.size), coin_psi, "-b.", label="coin. plan"
        )
        for traj_idx in range(n_select):
            ax.plot(
                range(idx + self.T_coin, idx + self.T_coin + cont_psi[traj_idx].size),
                cont_psi[traj_idx],
                "-.",
                color=traj_colors[traj_idx],
                label=f"cont. plan {traj_idx}",
            )
        ax.set_title("$\psi$ longitudinal angle, radians")
        ax.set_ylabel("rad")

    def __plot_planned_a(self, ax, frame_idx, planned_a):
        """Plot a, which is acceleration control input"""
        idx = frame_idx + 1
        ax.plot(
            range(idx, idx + planned_a.size),
            planned_a,
            "-.",
            color="orange",
            label="control plan",
        )
        ax.set_title("$a$ acceleration input, $m/s^2$")
        # ax.set_ylabel("$m/s^2$")

    def __plot_planned_coin_a(self, ax, traj_colors, frame_idx, coin_a, cont_a):
        """Plot a, which is acceleration control input"""
        idx = frame_idx + 1
        n_select = cont_a.shape[0]
        ax.plot(
            range(idx, idx + coin_a.size), coin_a, "-b.", label="coin. plan"
        )
        for traj_idx in range(n_select):
            ax.plot(
                range(
                    idx + self.T_coin - 1, idx + self.T_coin - 1 + cont_a[traj_idx].size
                ),
                cont_a[traj_idx],
                "-.",
                color=traj_colors[traj_idx],
                label=f"cont. plan {traj_idx}",
            )
        ax.set_title("$a$ acceleration input, $m/s^2$")

    def __plot_planned_delta(self, ax, frame_idx, planned_delta):
        """Plot delta, which is steering control input"""
        idx = frame_idx + 1
        ax.plot(
            range(idx, idx + planned_delta.size),
            planned_delta,
            "-.",
            color="orange",
            label="control plan",
        )
        ax.set_title("$\delta$ steering input, radians")
        # ax.set_ylabel("rad")

    def __plot_planned_coin_delta(
        self, ax, traj_colors, frame_idx, coin_delta, cont_delta
    ):
        """Plot delta, which is steering control input"""
        idx = frame_idx + 1
        n_select = cont_delta.shape[0]
        ax.plot(
            range(idx, idx + coin_delta.size), coin_delta, "-b.", label="coin. plan"
        )
        for traj_idx in range(n_select):
            ax.plot(
                range(
                    idx + self.T_coin - 1,
                    idx + self.T_coin - 1 + cont_delta[traj_idx].size,
                ),
                cont_delta[traj_idx],
                "-.",
                color=traj_colors[traj_idx],
                label=f"cont. plan {traj_idx}",
            )
        ax.set_title("$\delta$ steering input, radians")

    def plot_oa_timestep(
        self,
        frame_idx,
        planned_frame,
        planned_trajectory,
        planned_control,
        goal,
    ):
        """Helper function to plot one OA plan."""
        planned_xy = planned_trajectory[:, :2]
        planned_psi = planned_trajectory[:, 2]
        planned_v = planned_trajectory[:, 3]
        planned_a = planned_control[:, 0]
        planned_delta = planned_control[:, 1]
        actual_xy = self.actual_trajectory[:, :2]
        actual_psi = self.actual_trajectory[:, 2]
        actual_v = self.actual_trajectory[:, 3]
        min_x, min_y = np.min(planned_xy, axis=0)
        max_x, max_y = np.max(planned_xy, axis=0)
        x_mid, y_mid = (max_x + min_x) / 2, (max_y + min_y) / 2
        extent = (x_mid - PADDING, x_mid + PADDING, y_mid - PADDING, y_mid + PADDING)

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.ravel()

        ax = axes[0]  # BEV 1
        self.__render_scene_bev(ax, goal, extent)
        self.__plot_ov_trajectory(ax)
        self.__plot_ev_current_bbox(ax, frame_idx)
        self.__plot_ev_planned_trajectory(ax, planned_xy)
        self.__plot_ev_actual_trajectory(ax, frame_idx, actual_xy)

        ax = axes[1]  # BEV 2
        self.__plot_ev_planned_bbox(ax, planned_xy, planned_psi)
        self.__plot_ev_actual_bbox(ax, actual_xy, actual_psi)

        for ax in axes[:2]:
            self.__set_bev(ax, frame_idx, extent)

        # Plots of state/control
        self.__plot_v(axes[2], frame_idx, planned_v, actual_v)
        self.__plot_psi(axes[3], frame_idx, planned_psi, actual_psi)
        self.__plot_planned_a(axes[4], frame_idx, planned_a)
        self.__plot_planned_delta(axes[5], frame_idx, planned_delta)
        for ax in axes[2:]:
            ax.set_xlabel("timestep")
            ax.grid()
            ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join("out", f"{self.filename}_idx{frame_idx}.png"))
        plt.close(fig)

    def plot_oa(self):
        """Plot for original approach MPC plans over all simulation steps."""
        for (planned_frame, planned_trajectory, planned_control, goal) in zip(
            self.planned_frames,
            self.planned_trajectories,
            self.planned_controls,
            self.goals,
        ):
            frame_idx = np.argwhere(self.frames == planned_frame)[0, 0]
            self.plot_oa_timestep(
                frame_idx,
                planned_frame,
                planned_trajectory,
                planned_control,
                goal,
            )

    def plot_mcc_timestep(
        self,
        frame_idx,
        planned_frame,
        planned_trajectory,
        planned_control,
        goal,
    ):
        """Helper function to plot one (R)MCC plan."""

        # coinciding component of trajectory
        coin_trajectory = planned_trajectory[0, : (self.T_coin + 1)]
        coin_control = planned_control[0, : self.T_coin]
        coin_xy = coin_trajectory[:, :2]
        coin_psi = coin_trajectory[:, 2]
        coin_v = coin_trajectory[:, 3]
        coin_a = coin_control[:, 0]
        coin_delta = coin_control[:, 1]

        # contingency component of trajectory
        cont_trajectory = planned_trajectory[:, self.T_coin :]
        cont_control = planned_control[:, (self.T_coin - 1) :]
        cont_xy = cont_trajectory[..., :2]
        cont_psi = cont_trajectory[..., 2]
        cont_v = cont_trajectory[..., 3]
        cont_a = cont_control[..., 0]
        cont_delta = cont_control[..., 1]

        n_select = planned_trajectory.shape[0]
        actual_xy = self.actual_trajectory[..., :2]
        actual_psi = self.actual_trajectory[..., 2]
        actual_v = self.actual_trajectory[..., 3]
        _planned_xy = planned_trajectory[..., :2].reshape(-1, 2)
        min_x, min_y = np.min(_planned_xy, axis=0)
        max_x, max_y = np.max(_planned_xy, axis=0)
        x_mid, y_mid = (max_x + min_x) / 2, (max_y + min_y) / 2
        extent = (x_mid - PADDING, x_mid + PADDING, y_mid - PADDING, y_mid + PADDING)

        # Generate plots for map, state and inputs
        traj_colors = cm.winter(np.linspace(0, 1, n_select))
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.ravel()

        ax = axes[0]  # BEV 1
        self.__render_scene_bev(ax, goal, extent)
        self.__plot_ov_trajectory(ax)
        self.__plot_ev_current_bbox(ax, frame_idx)
        self.__plot_ev_planned_coin_trajectory(ax, traj_colors, coin_xy, cont_xy)
        self.__plot_ev_actual_trajectory(ax, frame_idx, actual_xy)

        ax = axes[1]  # BEV 2
        self.__plot_ev_planned_coin_bbox(
            ax, traj_colors, coin_xy, coin_psi, cont_xy, cont_psi
        )
        self.__plot_ev_actual_bbox(ax, actual_xy, actual_psi)

        for ax in axes[:2]:
            self.__set_bev(ax, frame_idx, extent)

        self.__plot_coin_v(
            axes[2], traj_colors, frame_idx, coin_v, cont_v, actual_v
        )
        self.__plot_coin_psi(
            axes[3], traj_colors, frame_idx, coin_psi, cont_psi, actual_psi
        )
        self.__plot_planned_coin_a(
            axes[4], traj_colors, frame_idx, coin_a, cont_a
        )
        self.__plot_planned_coin_delta(
            axes[5], traj_colors, frame_idx, coin_delta, cont_delta
        )
        for ax in axes[2:]:
            ax.set_xlabel("timestep")
            ax.grid()
            ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join("out", f"{self.filename}_idx{frame_idx}.png"))
        plt.close(fig)

    def plot_mcc(self):
        """Plot for (R)MCC contingency MPC plans over all simulation steps."""
        for (planned_frame, planned_trajectory, planned_control, goal) in zip(
            self.planned_frames,
            self.planned_trajectories,
            self.planned_controls,
            self.goals,
        ):
            frame_idx = np.argwhere(self.frames == planned_frame)[0, 0]
            self.plot_mcc_timestep(
                frame_idx,
                planned_frame,
                planned_trajectory,
                planned_control,
                goal,
            )


class PlotPIDController(object):

    def __init__(self, lowlevel, fixed_delta_seconds, filename="simulation"):
        """
        Parameters
        ==========
        lowlevel : util.AttrDict
            PID statistics.
        fixed_delta_seconds
            Simulator steptime given by carla.Settings.
        filename : str
            Label for plot.
        """
        self.lowlevel = lowlevel
        self.fixed_delta_seconds = fixed_delta_seconds
        self.filename = filename

    def __plot_speed(self, ax, steptimes, m_speed, r_speed):
        """Plot measured and reference speed."""
        ax.plot(steptimes, m_speed, "-g.", label="measured speed")
        ax.plot(steptimes, r_speed, "-", marker='.', color="orange", label="reference speed")
        ax.set_title("speed")
        ax.set_ylabel("m/s")

    def __plot_speed_error(self, ax, steptimes, pe_speed, ie_speed, de_speed):
        ax.plot(steptimes, pe_speed, "-r.", label="prop. error")
        ax.plot(steptimes, ie_speed, "-b.", label="integral error")
        ax.plot(steptimes, de_speed, "-g.", label="derivative error")
        ax.set_title("speed error")
        # ax.set_ylim([-10, 10])

    def __plot_lon_control(self, ax, steptimes, c_throttle, c_brake):
        """Plot longitudinal control."""
        ax.plot(steptimes, c_throttle, "-b.", label="applied throttle")
        ax.plot(steptimes, c_brake, "-r.", label="applied throttle")
        ax.set_title("longitudinal control")

    def __plot_angle(self, ax, steptimes, m_angle, r_angle):
        """Plot measured and reference angle."""
        ax.plot(steptimes, m_angle, "-g.", label="measured angle")
        ax.plot(steptimes, r_angle, "-", marker='.', color="orange", label="reference angle")
        ax.set_title("angle")
        ax.set_ylabel("rad")
    
    def __plot_angle_error(self, ax, steptimes, pe_angle, ie_angle, de_angle):
        ax.plot(steptimes, pe_angle, "-r.", label="prop. error")
        ax.plot(steptimes, ie_angle, "-b.", label="integral error")
        ax.plot(steptimes, de_angle, "-g.", label="derivative error")
        ax.set_title("angle error")
        # ax.set_ylim([-10, 10])

    def __plot_lat_control(self, ax, steptimes, c_steer):
        ax.plot(steptimes, c_steer, "-b.", label="applied steer")
        ax.set_title("lateral control")

    def plot(self):
        """Plot PID controller / actuation."""

        # extract the values.
        _, lowlevel = util.unzip([i for i in self.lowlevel.items()])
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

        # generate plot.
        fig, axes = plt.subplots(3, 2, figsize=(20, 20))
        axes = axes.T.ravel()
        steptimes = np.arange(len(lowlevel)) * self.fixed_delta_seconds
        self.__plot_speed(axes[0], steptimes, m_speed, r_speed)
        self.__plot_speed_error(axes[1], steptimes, pe_speed, ie_speed, de_speed)
        self.__plot_lon_control(axes[2], steptimes, c_throttle, c_brake)
        self.__plot_angle(axes[3], steptimes, m_angle, r_angle)
        self.__plot_angle_error(axes[4], steptimes, pe_angle, ie_angle, de_angle)
        self.__plot_lat_control(axes[5], steptimes, c_steer)
        for ax in axes:
            ax.grid()
            ax.legend()
            ax.set_xlabel("seconds s")
        fig.tight_layout()
        fig.savefig(os.path.join('out', f"{self.filename}_pid.png"))
        plt.close(fig)

class PlotCluster(object):

    def __init__(
        self,
        map_data,
        states,
        bboxes,
        vertices,
        OK_Ab_union,
        T,
        grid_shape="wide",
        filename="clusters"
    ):
        """
        map_data : util.AttrDict
            Container of vertices for road segments and lines produced by MapQuerier.
        states : collections.OrderedDict
            Collected vehicle states on when data collector makes a prediction.
        bboxes : collections.OrderedDict
            Collected vehicle bounding boxes when data collector makes a prediction.
        vertices : collections.OrderedDict
            For each frame, the vertices are indexed by (T, max(K), O).
            Vertex set has shape (N,4,2).
        OK_Ab_union : collections.OrderedDict
            For each frame, the A, b are indexed by (T, max(K), O).
            A has shape (L, 2) b has shape (L,).
        T : int
            Prediction horizon.
        """
        self.map_data = map_data
        frames, states = util.unzip([i for i in states.items()])
        self.frames = np.array(frames)
        # TODO: don't stack states and bboxes. Varying numbers of OVs
        # self.states = np.stack(states)
        # self.bboxes = np.stack(bboxes.values())
        self.states = list(states)
        self.bboxes = list(bboxes.values())
        self.vertices = list(vertices.values())
        self.OK_Ab_union = list(OK_Ab_union.values())
        self.T = T
        self.filename = filename
        self.grid_shape = grid_shape

    def __create_plot_grid(self, N):
        """Plots for paper."""
        if self.grid_shape == "tall":
            # make the plotting grid tall
            fig, axes = plt.subplots(
                N // 2 + (N % 2), 2, figsize=(10, (10 / 4) * N)
            )
        elif self.grid_shape == "wide":
            # make the plotting grid wide
            fig, axes = plt.subplots(
                2, N // 2 + (N % 2), figsize=((10 / 4) * N, 10)
            )
        else:
            NotImplementedError(f"Unknown grid shape {self.grid_shape}")
        return fig, axes

    def __vertices_to_extent(self, vertices):
        x_min, y_min = np.min(vertices, axis=0) - 5
        x_max, y_max = np.max(vertices, axis=0) + 5
        return x_min, x_max, y_min, y_max

    def __render_scene_bev(self, ax, extent):
        render_map_crop(ax, self.map_data, extent)

    def __collect_aggregate_predictions(self, n):
        """Extract aggregate predictions t|t' over all timesteps t' < t for timestep t."""
        vertices_acrosstime = []
        O_acrosstime = []
        K_acrosstime = []
        A_union_acrosstime = []
        b_union_acrosstime = []
        for t in range(self.T):
            m = n - t - 1
            if m < 0:
                break
            vertices_acrosstime.append(self.vertices[m][t])
            O_acrosstime.append(self.OK_Ab_union[m][0])
            K_acrosstime.append(self.OK_Ab_union[m][1])
            A_union_acrosstime.append(self.OK_Ab_union[m][2][t])
            b_union_acrosstime.append(self.OK_Ab_union[m][3][t])
        return (
            vertices_acrosstime,
            O_acrosstime,
            K_acrosstime,
            A_union_acrosstime,
            b_union_acrosstime,
        )

    def __make_colors(self):
        return util.map_to_list(lambda cmap: cmap(np.linspace(0, 1, self.T)), OVEHICLE_COLORS)

    def __render_vehicle_bbox(self, ax, n, o, color):
        state = self.states[n][o]
        bbox  = self.bboxes[n][o]
        position = state[:2]
        heading  = state[2]
        vertices = npu.vertices_from_bbox(position, heading, bbox)
        bb = patches.Polygon(
            vertices.reshape((-1, 2,))  # fmt: skip
            ,
            closed=True,
            color=color,
            fc=util.plu.modify_alpha(color, 0.2),
            ls="-",
        )
        ax.add_patch(bb)

    def plot_overapprox_per_timestep(self, n):
        ovehicle_colors = self.__make_colors()
        (
            vertices_acrosstime,
            O_acrosstime,
            K_acrosstime,
            A_union_acrosstime,
            b_union_acrosstime,
        ) = self.__collect_aggregate_predictions(n)
        O_max = max(O_acrosstime)
        fig, axes = self.__create_plot_grid(O_max)
        axes = axes.ravel()

        cmap = cm.hsv
        norm = clr.Normalize(vmin=1, vmax=self.T + 1, clip=True)
        overapprox_colors = cmap(np.linspace(0, 1, self.T + 1))
        for ax, o in zip(axes, range(O_max)):
            vertices = util.select_from_nested_list_at_levelindex(vertices_acrosstime, 2, o)
            vertices = util.filter_to_list(lambda a: a is not None, vertices)
            extent = self.__vertices_to_extent(
                np.concatenate(vertices).reshape((-1, 2))
            )
            self.__render_scene_bev(ax, extent)
            # color = ovehicle_colors[o][0]
            # self.__render_vehicle_bbox(ax, n, o, color)
            self.__render_vehicle_bbox(ax, n, o, "black")

            for t in range(self.T):
                try:
                    K_acrosstime[t][o]
                except IndexError:
                    continue
                color = overapprox_colors[t]
                for k in range(K_acrosstime[t][o]):
                    A = A_union_acrosstime[t][k][o]
                    b = b_union_acrosstime[t][k][o]
                    npu.plot_h_polyhedron(ax, A, b, ec=color, lw=2, alpha=0.7)

            ax.set_title(f"vehicle {o}")
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        fig.tight_layout()
        fig.savefig(os.path.join("out", f"{self.filename}_n{n}.png"))
        plt.close(fig)

    def plot_overapprox_per_vehicle(self):
        N = len(self.frames)
        for n in range(1, N):
            self.plot_overapprox_per_timestep(n)

    def plot_convexhull_per_timestep(self, n):
        ovehicle_colors = self.__make_colors()
        (
            vertices_acrosstime,
            O_acrosstime,
            K_acrosstime,
            A_union_acrosstime,
            b_union_acrosstime,
        ) = self.__collect_aggregate_predictions(n)
        O_max = max(O_acrosstime)
        fig, axes = self.__create_plot_grid(O_max)
        axes = axes.ravel()

        cmap = cm.hsv
        norm = clr.Normalize(vmin=1, vmax=self.T + 1, clip=True)
        overapprox_colors = cmap(np.linspace(0, 1, self.T + 1))
        for ax, o in zip(axes, range(O_max)):
            vertices = util.select_from_nested_list_at_levelindex(vertices_acrosstime, 2, o)
            vertices = util.filter_to_list(lambda a: a is not None, vertices)
            extent = self.__vertices_to_extent(
                np.concatenate(vertices).reshape((-1, 2))
            )
            self.__render_scene_bev(ax, extent)
            # color = ovehicle_colors[o][0]
            # self.__render_vehicle_bbox(ax, n, o, color)
            self.__render_vehicle_bbox(ax, n, o, "black")

            for t in range(self.T):
                try:
                    K_acrosstime[t][o]
                except IndexError:
                    continue
                color = overapprox_colors[t]
                for k in range(K_acrosstime[t][o]):
                    vertices = np.reshape(vertices_acrosstime[t][k][o], (-1, 2))
                    ch = scipy.spatial.ConvexHull(vertices, incremental=False)
                    for simplex in ch.simplices:
                        ax.plot(
                            vertices[simplex, 0], vertices[simplex, 1],
                            '-', color=color, lw=2, alpha=0.7
                        )

            ax.set_title(f"vehicle {o}")
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        fig.tight_layout()
        fig.savefig(os.path.join("out", f"{self.filename}_n{n}.png"))
        plt.close(fig)

    def plot_convexhull_per_vehicle(self):
        N = len(self.frames)
        for n in range(1, N):
            self.plot_convexhull_per_timestep(n)

    def plot_timestep(self, ax, n):
        """Plot clusters
        
        Example predictions on timesteps 10, 11, 12, 13, 14
        10 : 11|10 12|10 13|10 14|10
        11 : 12|11 13|11 14|11 15|11
        12 : 13|12 14|12 15|12 16|12
        13 : 14|13 15|13 16|13 17|13
        14 : 15|14 16|14 17|14 18|14
        """
        ovehicle_colors = self.__make_colors()
        (
            vertices_acrosstime,
            O_acrosstime,
            K_acrosstime,
            A_union_acrosstime,
            b_union_acrosstime,
        ) = self.__collect_aggregate_predictions(n)

        # Render scene
        vertices = util.flatten_nested_list(vertices_acrosstime, include=np.ndarray)
        extent = self.__vertices_to_extent(
            np.concatenate(vertices).reshape((-1, 2))
        )
        self.__render_scene_bev(ax, extent)

        # Render vehicle bounding boxes of current timestep
        O = self.OK_Ab_union[n][0]
        for o in range(O):
            color = ovehicle_colors[o][0]
            self.__render_vehicle_bbox(ax, n, o, color)

        # Render outer approximations
        for t in range(self.T):
            if t >= len(O_acrosstime):
                break
            for o in range(O_acrosstime[t]):
                # for vehicle o
                for k in range(K_acrosstime[t][o]):
                    # for vehicle latent k of vehicle o
                    color = ovehicle_colors[o][t]
                    A = A_union_acrosstime[t][k][o]
                    b = b_union_acrosstime[t][k][o]
                    npu.plot_h_polyhedron(ax, A, b, ec=color, ls="-", alpha=1)

    def plot(self):
        N = len(self.frames)
        fig, axes = plt.subplots(N // 2 + (N % 2), 2, figsize=(10, (10 / 4)*N))
        axes = axes.ravel()
        for n, ax in zip(range(1, N), axes):
            self.plot_timestep(ax, n)
        fig.tight_layout()
        fig.savefig(os.path.join("out", f"{self.filename}.png"))
        plt.close(fig)
