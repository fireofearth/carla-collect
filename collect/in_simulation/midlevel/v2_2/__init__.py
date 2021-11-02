"""
v2 is a generalization of v1.

    - This is the original approach motion planner.
    - It clusters predictions from other vehicles and computes a trajectory using a double integrator model for the ego vehicle.
    - The dynamics are adapted based on the vehicle's heading at each iteration of the MPC.

v2_1 has road boundary conditions, closed loop.
Code profiling is also done here.
v2_2 has some refactoring, and cleaner code.
"""

# Built-in libraries
import sys
import os
import math
import numbers
import logging
import collections
import weakref
import copy

# PyPI libraries
import numpy as np
import scipy.spatial
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm
import matplotlib.patches as patches
import torch
import control
import control.matlab
import docplex.mp
import docplex.mp.model

# Local libraries
import carla
import utility as util
import carlautil
import carlautil.debug

try:
    from utils.trajectory_utils import prediction_output_to_trajectories
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from ..util import (get_vertices_from_center, profile,
        get_approx_union, plot_h_polyhedron, get_ovehicle_color_set,
        plot_lcss_prediction, get_vertices_from_centers,
        plot_oa_simulation)
from ..ovehicle import OVehicle
from ..prediction import generate_vehicle_latents
from ...lowlevel import LocalPlanner
from ....generate import AbstractDataCollector
from ....generate import create_semantic_lidar_blueprint
from ....generate.map import NaiveMapQuerier
from ....generate.scene import OnlineConfig, SceneBuilder
from ....generate.scene.v2_2.trajectron_scene import (
        TrajectronPlusPlusSceneBuilder)
from ....generate.scene.trajectron_util import (
        standardization, plot_trajectron_scene)

class MidlevelAgent(AbstractDataCollector):
    """Controller for vehicle using predictions."""

    Z_SENSOR_REL = 2.5

    def __create_segmentation_lidar_sensor(self):
        return self.__world.spawn_actor(
                create_semantic_lidar_blueprint(self.__world),
                carla.Transform(carla.Location(z=self.Z_SENSOR_REL)),
                attach_to=self.__ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid)

    @staticmethod
    def __get_state_space_representation(prediction_timestep):
        """Get state-space representation of double integrator model.
        """
        # A, sys.A both have shape (4, 4)
        A = np.diag([1, 1], k=2)
        # B, sys.B both have shape (4, 2)
        B = np.concatenate((np.diag([0,0]), np.diag([1,1]),))
        # C has shape (2, 4)
        C = np.concatenate((np.diag([1,1]), np.diag([0,0]),), axis=1)
        # D has shape (2, 2)
        D = np.diag([0, 0])
        sys = control.matlab.c2d(control.matlab.ss(A, B, C, D), prediction_timestep)
        A = np.array(sys.A)
        B = np.array(sys.B)
        return (A, B)
    
    def __make_global_params(self):
        """Get Global LCSS parameters used across all loops"""
        params = util.AttrDict()
        params.M_big = 1000
        params.u_max = 3.
        params.A, params.B = self.__get_state_space_representation(
                self.__prediction_timestep)
        # number of state variables x, number of input variables u
        # nx = 4, nu = 2
        params.nx, params.nu = params.B.shape
        bbox_lon, bbox_lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
        params.diag = np.sqrt(bbox_lon**2 + bbox_lat**2) / 2.
        # Prediction parameters
        params.T = self.__prediction_horizon
        params.L = 4 # number of faces of obstacle sets

        # Closed for solution of control without obstacles
        A, B, T, nx, nu = params.A, params.B, params.T, params.nx, params.nu
        # C1 has shape (nx, T*nx)
        C1 = np.zeros((nx, T*nx,))
        # C2 has shape (nx*(T - 1), nx*(T-1)) as A has shape (nx, nx)
        C2 = np.kron(np.eye(T - 1), A)
        # C3 has shape (nx*(T - 1), nx)
        C3 = np.zeros(((T - 1)*nx, nx,))
        # C, Abar have shape (nx*T, nx*T)
        C = np.concatenate((C1, np.concatenate((C2, C3,), axis=1),), axis=0)
        Abar = np.eye(T * nx) - C
        # Bbar has shape (nx*T, nu*T) as B has shape (nx, nu)
        Bbar = np.kron(np.eye(T), B)
        # Gamma has shape (nx*(T + 1), nu*T) as Abar\Bbar has shape (nx*T, nu*T)
        Gamma = np.concatenate((np.zeros((nx, T*nu,)),
                np.linalg.solve(Abar, Bbar),))
        params.Abar = Abar
        params.Bbar = Bbar
        params.Gamma = Gamma
        return params

    def __init__(self,
            ego_vehicle,
            map_reader,
            other_vehicle_ids,
            eval_stg,
            control_horizon=6,
            n_burn_interval=4,
            prediction_horizon=8,
            n_predictions=100,
            scene_builder_cls=TrajectronPlusPlusSceneBuilder,
            scene_config=OnlineConfig(),
            log_cplex=False,
            plot_scenario=False,
            plot_simulation=False,
            plot_boundary=False,
            plot_vertices=False,
            plot_overapprox=False,
            **kwargs):
        assert control_horizon <= prediction_horizon
        self.__ego_vehicle = ego_vehicle
        self.__map_reader = map_reader
        self.__world = self.__ego_vehicle.get_world()

        # __first_frame : int
        #   First frame in simulation. Used to find current timestep.
        self.__first_frame = None
        self.__scene_builder = None
        self.__scene_config = scene_config
        self.__scene_builder_cls = scene_builder_cls
        # __control_horizon : int
        #   Number of predictions timesteps to conduct control over.
        self.__control_horizon = control_horizon
        # __n_burn_interval : int
        #   Interval in prediction timesteps to skip prediction and control.
        self.__n_burn_interval = n_burn_interval
        # __prediction_horizon : int
        #   Number of predictions timesteps to predict other vehicles over.
        self.__prediction_horizon = prediction_horizon
        self.__n_predictions = n_predictions
        self.__eval_stg = eval_stg

        vehicles = self.__world.get_actors(other_vehicle_ids)
        # __other_vehicles : list of carla.Vehicle
        #     List of IDs of vehicles not including __ego_vehicle.
        #     Use this to track other vehicles in the scene at each timestep. 
        self.__other_vehicles = dict(zip(other_vehicle_ids, vehicles))

        # __sensor : carla.Sensor
        #     Segmentation sensor. Data points will be used to construct overhead.
        self.__sensor = self.__create_segmentation_lidar_sensor()

        # __lidar_feeds : collections.OrderedDict
        #     Where int key is frame index and value
        #     is a carla.LidarMeasurement or carla.SemanticLidarMeasurement
        self.__lidar_feeds = collections.OrderedDict()

        self.__prediction_timestep = self.__scene_config.record_interval \
                * self.__world.get_settings().fixed_delta_seconds
        self.__local_planner = LocalPlanner(self.__ego_vehicle)
        self.__params = self.__make_global_params()
        self.__goal = util.AttrDict(x=50, y=0, is_relative=True)
        # __road_segment_enclosure : np.array
        #   Array of shape (4, 2) enclosing the road segment
        # __road_seg_starting : np.array
        #   The position and the heading angle of the starting waypoint
        #   of the road of form [x, y, angle] in (meters, meters, radians).
        self.__road_seg_starting, self.__road_seg_enclosure, self.__road_seg_params \
                = self.__map_reader.road_segment_enclosure_from_actor(self.__ego_vehicle)
        self.__road_seg_starting[1] *= -1 # need to flip about x-axis
        self.__road_seg_starting[2] = util.reflect_radians_about_x_axis(
                self.__road_seg_starting[2]) # need to flip about x-axis
        self.__road_seg_enclosure[:, 1] *= -1 # need to flip about x-axis
        self.U_warmstarting = None

        self.log_cplex       = log_cplex
        self.plot_scenario   = plot_scenario
        self.plot_simulation = plot_simulation
        self.plot_boundary   = plot_boundary
        self.plot_vertices   = plot_vertices
        self.plot_overapprox = plot_overapprox

        if self.plot_simulation:
            self.__plot_simulation_data = util.AttrDict(
                actual_trajectory=collections.OrderedDict(),
                planned_trajectories=collections.OrderedDict(),
                planned_controls=collections.OrderedDict()
            )
    
    def get_vehicle_state(self):
        """Get the vehicle state as an ndarray. State consists of
        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z,
        length, width, height, pitch, yaw, roll] where pitch, yaw, roll are in
        radians."""
        return carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(self.__ego_vehicle)

    def get_goal(self):
        return copy.copy(self.__goal)

    def set_goal(self, x, y, is_relative=True):
        self.__goal = util.AttrDict(x=x, y=y, is_relative=is_relative)

    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.__sensor.listen(lambda image: type(self).parse_image(weak_self, image))
    
    def stop_sensor(self):
        """Stop the sensor."""
        self.__sensor.stop()

    @property
    def sensor_is_listening(self):
        return self.__sensor.is_listening

    def __plot_simulation(self):
        if len(self.__plot_simulation_data.planned_trajectories) == 0:
            return
        filename = f"agent{self.__ego_vehicle.id}_oa_simulation"
        lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
        plot_oa_simulation(
            self.__scene_builder.get_scene(),
            self.__plot_simulation_data.actual_trajectory,
            self.__plot_simulation_data.planned_trajectories,
            self.__plot_simulation_data.planned_controls,
            [lon, lat],
            self.__control_horizon,
            filename=filename
        )

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.__sensor.destroy()
        self.__sensor = None
        if self.plot_simulation:
            self.__plot_simulation()

    def do_prediction(self, frame):
        """Get processed scene object from scene builder, input the scene to a model to
        generate the predictions, and then return the predictions and the latents variables.
        """

        """Construct online scene"""
        scene = self.__scene_builder.get_scene()

        """Extract Predictions"""
        frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
        timestep = frame_id # we use this as the timestep
        timesteps = np.array([timestep])
        with torch.no_grad():
            z, predictions, nodes, predictions_dict, latent_probs = generate_vehicle_latents(
                    self.__eval_stg, scene, timesteps,
                    num_samples=self.__n_predictions,
                    ph=self.__prediction_horizon,
                    z_mode=False, gmm_mode=False, full_dist=False, all_z_sep=False)

        _, past_dict, ground_truth_dict = \
                prediction_output_to_trajectories(
                    predictions_dict, dt=scene.dt, max_h=10,
                    ph=self.__prediction_horizon, map=None)
        return util.AttrDict(scene=scene, timestep=timestep, nodes=nodes,
                predictions=predictions, z=z, latent_probs=latent_probs,
                past_dict=past_dict, ground_truth_dict=ground_truth_dict)
        
    def make_ovehicles(self, result):
        scene, timestep, nodes = result.scene, result.timestep, result.nodes
        predictions, latent_probs, z = result.predictions, result.latent_probs, result.z
        past_dict, ground_truth_dict = result.past_dict, result.ground_truth_dict

        """Preprocess predictions"""
        minpos = np.array([scene.x_min, scene.y_min])
        ovehicles = []
        for idx, node in enumerate(nodes):
            if node.id == 'ego':
                continue
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__other_vehicles[int(node.id)])
            veh_bbox = [lon, lat]
            veh_gt         = ground_truth_dict[timestep][node] + minpos
            veh_past       = past_dict[timestep][node] + minpos
            veh_predict    = predictions[idx] + minpos
            veh_latent_pmf = latent_probs[idx]
            n_states = veh_latent_pmf.size
            zn = z[idx]
            veh_latent_predictions = [[] for x in range(n_states)]
            for jdx, p in enumerate(veh_predict):
                veh_latent_predictions[zn[jdx]].append(p)
            for jdx in range(n_states):
                veh_latent_predictions[jdx] = np.array(veh_latent_predictions[jdx])
            ovehicle = OVehicle.from_trajectron(node,
                    self.__prediction_horizon, veh_gt, veh_past,
                    veh_latent_pmf, veh_latent_predictions, bbox=veh_bbox)
            ovehicles.append(ovehicle)
        
        return ovehicles

    def make_local_params(self, ovehicles):
        """Get Local LCSS parameters that are environment dependent."""
        params = util.AttrDict()
        # O - number of obstacles
        params.O = len(ovehicles)
        # K - for each o=1,...,O K[o] is the number of outer approximations for vehicle o
        params.K = np.zeros(params.O, dtype=int)
        for idx, vehicle in enumerate(ovehicles):
            params.K[idx] = vehicle.n_states

        p_0_x, p_0_y, _ = carlautil.actor_to_location_ndarray(
                self.__ego_vehicle, flip_y=True)
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(
                self.__ego_vehicle, flip_y=True)
        # x0 : np.array
        #   Initial state
        x0 = np.array([p_0_x, p_0_y, v_0_x, v_0_y])
        params.x0 = x0
        A, T = self.__params.A, self.__params.T
        # States_free_init has shape (nx*(T+1))
        params.States_free_init = np.concatenate([
                np.linalg.matrix_power(A, t) @ x0 for t in range(T+1)])
        return params

    def __plot_boundary(self):
        fig, axes = plt.subplots(1, 2, figsize=(10,6))
        axes = axes.ravel()
        ego_x, ego_y, _ = carlautil.to_location_ndarray(self.__ego_vehicle)
        ego_y = -ego_y
        wp_x, wp_y = self.__road_seg_starting[:2]
        axes[0].plot(ego_x, ego_y, 'g*')
        axes[0].plot(wp_x, wp_y, 'b*')
        patch = patches.Polygon(self.__road_seg_enclosure, ec='b', fc='none')
        axes[0].add_patch(patch)
        axes[0].set_aspect('equal')
        b_length, f_length, r_width, l_width = self.__road_seg_params
        mtx = util.rotation_2d(self.__road_seg_starting[2])
        shift = self.__road_seg_starting[:2]
        ego_x, ego_y = mtx @ (np.array([ego_x, ego_y]) - shift)
        axes[1].plot(ego_x, ego_y, 'g*')
        axes[1].plot([0, -b_length], [0, 0], '-bo')
        axes[1].plot([0,  f_length], [0, 0], '-bo')
        axes[1].plot([0,0], [0, -r_width], '-bo')
        axes[1].plot([0,0], [0,  l_width], '-bo')
        axes[1].set_aspect('equal')
        fig.tight_layout()
        plt.show()

    def compute_boundary_constraints(self, p_x, p_y):
        """
        Parameters
        ==========
        p_x : np.array of docplex.mp.vartype.VarType
        p_y : np.array of docplex.mp.vartype.VarType
        """
        b_length, f_length, r_width, l_width = self.__road_seg_params
        mtx = util.rotation_2d(self.__road_seg_starting[2])
        shift = self.__road_seg_starting[:2]

        if self.plot_boundary:
            self.__plot_boundary()

        pos = np.stack([p_x, p_y], axis=1)
        pos = util.obj_matmul((pos - shift), mtx.T)
        constraints = []
        constraints.extend([-b_length <= z             for z in pos[:, 0]])
        constraints.extend([             z <= f_length for z in pos[:, 0]])
        constraints.extend([-r_width  <= z             for z in pos[:, 1]])
        constraints.extend([             z <= l_width  for z in pos[:, 1]])
        return constraints

    def compute_velocity_constraints(self, v_x, v_y):
        """Velocity states have coupled constraints.
        Generate docplex constraints for velocity for double integrators.

        Street speed limit is 30 km/h == 8.33.. m/s

        Parameters
        ==========
        v_x : np.array of docplex.mp.vartype.VarType
        v_y : np.array of docplex.mp.vartype.VarType
        """
        v_lim = self.__ego_vehicle.get_speed_limit() # is m/s
        _, theta, _ = carlautil.actor_to_rotation_ndarray(
                self.__ego_vehicle, flip_y=True)
        r = v_lim / 2
        v_1 = r
        v_2 = 0.75 * v_lim
        c1 = v_2*((v_x - r*np.cos(theta))*np.cos(theta) \
                + (v_y - r*np.sin(theta))*np.sin(theta))
        c2 = v_1*((v_y - r*np.sin(theta))*np.cos(theta) \
                - (v_x - r*np.cos(theta))*np.sin(theta))
        c3 = np.abs(v_1 * v_2)
        constraints = []
        constraints.extend([ z <= c3 for z in  c1 + c2 ])
        constraints.extend([ z <= c3 for z in -c1 + c2 ])
        constraints.extend([ z <= c3 for z in  c1 - c2 ])
        constraints.extend([ z <= c3 for z in -c1 - c2 ])
        return constraints

    def compute_acceleration_constraints(self, u_x, u_y):
        """Accelaration control inputs have coupled constraints.
        Generate docplex constraints for velocity for double integrators.

        Present performance cars are capable of going from 0 to 60 mph in under 5 seconds.
        Reference:
        https://en.wikipedia.org/wiki/0_to_60_mph

        Parameters
        ==========
        u_x : np.array of docplex.mp.vartype.VarType
        u_y : np.array of docplex.mp.vartype.VarType
        """
        u_max = self.__params.u_max
        _, theta, _ = carlautil.actor_to_rotation_ndarray(
                self.__ego_vehicle, flip_y=True)
        r = -u_max*(1. / 2.)
        a_1 = u_max*(3. / 2.)
        a_2 = u_max
        c1 = a_2*((u_x - r*np.cos(theta))*np.cos(theta) \
                + (u_y - r*np.sin(theta))*np.sin(theta))
        c2 = a_1*((u_y - r*np.sin(theta))*np.cos(theta) \
                - (u_x - r*np.cos(theta))*np.sin(theta))
        c3 = np.abs(a_1 * a_2)
        constraints = []
        constraints.extend([ z <= c3 for z in  c1 + c2 ])
        constraints.extend([ z <= c3 for z in -c1 + c2 ])
        constraints.extend([ z <= c3 for z in  c1 - c2 ])
        constraints.extend([ z <= c3 for z in -c1 - c2 ])
        return constraints

    def __plot_vertices(self, ovehicles, vertices):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        ax = axes[1]
        X = vertices[-1][0][0][:,0:2].T
        ax.scatter(X[0], X[1], c='r', s=2)
        X = vertices[-1][0][0][:,2:4].T
        ax.scatter(X[0], X[1], c='b', s=2)
        X = vertices[-1][0][0][:,4:6].T
        ax.scatter(X[0], X[1], c='g', s=2)
        X = vertices[-1][0][0][:,6:8].T
        ax.scatter(X[0], X[1], c='m', s=2)

        x, y, _ = carlautil.actor_to_location_ndarray(
                self.__ego_vehicle, flip_y=True)
        ax = axes[0]
        ax.scatter(x, y, marker='*', c='k', s=100)
        ovehicle_colors = get_ovehicle_color_set([ov.n_states for ov in ovehicles])
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                logging.info(f"Plotting OV {ov_idx} latent value {latent_idx}.")
                color = ovehicle_colors[ov_idx][latent_idx]
                for t in range(self.__params.T):
                    X = vertices[t][latent_idx][ov_idx][:,0:2].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:,2:4].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:,4:6].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:,6:8].T
                    ax.scatter(X[0], X[1], color=color, s=2)

        for ax in axes:
            ax.set_aspect('equal')
        plt.show()

    def __compute_vertices(self, params, ovehicles):
        """Compute verticles from predictions."""
        T, K, n_ov = self.__params.T, params.K, params.O
        vertices = np.empty((T, np.max(K), n_ov), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    ps = ovehicle.pred_positions[latent_idx][:,t]
                    yaws = ovehicle.pred_yaws[latent_idx][:,t]
                    vertices[t][latent_idx][ov_idx] = get_vertices_from_centers(
                            ps, yaws, ovehicle.bbox)

        if self.plot_vertices: self.__plot_vertices(ovehicles, vertices)
        return vertices
    
    def __plot_overapproximations(self, ovehicles, vertices, A_union, b_union):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        ax = axes[1]
        X = vertices[-1][0][0][:,0:2].T
        ax.scatter(X[0], X[1], color='r', s=2)
        X = vertices[-1][0][0][:,2:4].T
        ax.scatter(X[0], X[1], color='b', s=2)
        X = vertices[-1][0][0][:,4:6].T
        ax.scatter(X[0], X[1], color='g', s=2)
        X = vertices[-1][0][0][:,6:8].T
        ax.scatter(X[0], X[1], color='m', s=2)
        A = A_union[-1][0][0]
        b = b_union[-1][0][0]
        try:
            plot_h_polyhedron(ax, A, b, fc='none', ec='k')
        except scipy.spatial.qhull.QhullError as e:
            print(f"Failed to plot polyhedron of OV")

        x, y, _ = carlautil.actor_to_location_ndarray(
                self.__ego_vehicle, flip_y=True)
        ax = axes[0]
        ax.scatter(x, y, marker='*', c='k', s=100)
        ovehicle_colors = get_ovehicle_color_set([ov.n_states for ov in ovehicles])
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                logging.info(f"Plotting OV {ov_idx} latent value {latent_idx}.")
                color = ovehicle_colors[ov_idx][latent_idx]
                for t in range(self.__params.T):
                    X = vertices[t][latent_idx][ov_idx][:,0:2].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:,2:4].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:,4:6].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    X = vertices[t][latent_idx][ov_idx][:,6:8].T
                    ax.scatter(X[0], X[1], color=color, s=2)
                    A = A_union[t][latent_idx][ov_idx]
                    b = b_union[t][latent_idx][ov_idx]
                    try:
                        plot_h_polyhedron(ax, A, b, fc='none', ec=color)#, alpha=0.3)
                    except scipy.spatial.qhull.QhullError as e:
                        print(f"Failed to plot polyhedron of OV {ov_idx} latent value {latent_idx} timestep t={t}")
        
        for ax in axes:
            ax.set_aspect('equal')
        plt.show()

    def __compute_overapproximations(self, vertices, params, ovehicles):
        """Compute the approximation of the union of obstacle sets"""
        T, K, n_ov = self.__params.T, params.K, params.O
        A_union = np.empty((T, np.max(K), n_ov,), dtype=object).tolist()
        b_union = np.empty((T, np.max(K), n_ov,), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    yaws = ovehicle.pred_yaws[latent_idx][:,t]
                    vertices_k = vertices[t][latent_idx][ov_idx]
                    mean_theta_k = np.mean(yaws)
                    A_union_k, b_union_k = get_approx_union(mean_theta_k, vertices_k)
                    A_union[t][latent_idx][ov_idx] = A_union_k
                    b_union[t][latent_idx][ov_idx] = b_union_k
        
        """Plot the overapproximation"""
        if self.plot_overapprox:
            self.__plot_overapproximations(ovehicles, vertices, A_union, b_union)
        return A_union, b_union

    def do_highlevel_control(self, params, ovehicles):
        """Decide parameters"""
        # TODO: assign eps^k_o and beta^k_o to the vehicles.
        # Skipping that for now.
    
        vertices = self.__compute_vertices(params, ovehicles)
        A_union, b_union = self.__compute_overapproximations(vertices, params, ovehicles)
        
        """Apply motion planning problem"""
        L, T, K, Gamma, nu, nx = self.__params.L, self.__params.T, params.K, \
                self.__params.Gamma, self.__params.nu, self.__params.nx
        u_max = self.__params.u_max
        model = docplex.mp.model.Model(name="proposed_problem")
        u = np.array(model.continuous_var_list(nu*T, lb=-u_max, ub=u_max, name='u'),
                dtype=object)
        Delta = np.array(model.binary_var_list(L*np.sum(K)*T, name='delta'),
                dtype=object).reshape(np.sum(K), T, L)
        
        X = (params.States_free_init + util.obj_matmul(Gamma, u)).reshape(T + 1, nx)
        X = X[1:]
        U = u.reshape(T, nu)

        """Apply motion constraints"""
        model.add_constraints(self.compute_boundary_constraints(X[:, 0], X[:, 1]))
        model.add_constraints(self.compute_velocity_constraints(X[:, 2], X[:, 3]))
        model.add_constraints(self.compute_acceleration_constraints(U[:, 0], U[:, 1]))

        """Apply collision constraints"""
        T, K, diag, M_big = self.__params.T, params.K, \
                self.__params.diag, self.__params.M_big
        for ov_idx, ovehicle in enumerate(ovehicles):
            n_states = ovehicle.n_states
            sum_clu = np.sum(K[:ov_idx])
            for latent_idx in range(n_states):
                for t in range(T):
                    A_obs = A_union[t][latent_idx][ov_idx]
                    b_obs = b_union[t][latent_idx][ov_idx]
                    indices = sum_clu + latent_idx
                    lhs = util.obj_matmul(A_obs, X[t, :2]) + M_big*(1 - Delta[indices, t])
                    rhs = b_obs + diag
                    model.add_constraints([l >= r for (l,r) in zip(lhs, rhs)])
                    model.add_constraint(np.sum(Delta[indices, t]) >= 1)
        
        """Start from current vehicle position and minimize the objective"""
        p_x, p_y, _ = carlautil.actor_to_location_ndarray(
                self.__ego_vehicle, flip_y=True)
        if self.__goal.is_relative:
            goal_x, goal_y = p_x + self.__goal.x, p_y + self.__goal.y
        else:
            goal_x, goal_y = self.__goal.x, self.__goal.y
        start = np.array([p_x, p_y])
        goal = np.array([goal_x, goal_y])
        cost = (X[-1, 0] - goal_x)**2 + (X[-1, 1] - goal_y)**2
        model.minimize(cost)
        if self.U_warmstarting is not None:
            # Warm start inputs if past iteration was run.
            warm_start = model.new_solution()
            for i, u in enumerate(self.U_warmstarting[self.__control_horizon:]):
                warm_start.add_var_value(f"u_{2*i}", u[0])
                warm_start.add_var_value(f"u_{2*i + 1}", u[1])
            # add delta_0 as hotfix to MIP warmstart as it needs
            # at least 1 integer value set.
            warm_start.add_var_value('delta_0', 0)
            model.add_mip_start(warm_start)

        # model.print_information()
        # model.parameters.read.datacheck = 1
        if self.log_cplex:
            model.parameters.mip.display = 2
            s = model.solve(log_output=True)
        else:
            model.solve()
        # model.print_solution()

        f = lambda x: x if isinstance(x, numbers.Number) else x.solution_value
        cost = cost.solution_value
        U_star = util.obj_vectorize(f, U)
        X_star = util.obj_vectorize(f, X)
        return util.AttrDict(cost=cost, U_star=U_star, X_star=X_star,
                A_union=A_union, b_union=b_union, vertices=vertices,
                start=start, goal=goal)

    # @profile(sort_by='cumulative', lines_to_print=50, strip_dirs=True)
    def __compute_prediction_controls(self, frame):
        pred_result = self.do_prediction(frame)
        ovehicles = self.make_ovehicles(pred_result)
        params = self.make_local_params(ovehicles)
        ctrl_result = self.do_highlevel_control(params, ovehicles)

        """use control input next round for warm starting."""
        self.U_warmstarting = ctrl_result.U_star

        """Get trajectory"""
        trajectory = []
        x, y, _ = carlautil.to_location_ndarray(self.__ego_vehicle, flip_y=True)
        X = np.concatenate((np.array([x, y])[None], ctrl_result.X_star[:, :2]))
        n_steps = X.shape[0]
        headings = []
        for t in range(1, n_steps):
            x, y = X[t]
            y = -y # flip about x-axis again to move back to UE coordinates
            yaw = np.arctan2(X[t, 1] - X[t - 1, 1], X[t, 0] - X[t - 1, 0])
            headings.append(yaw)
             # flip about x-axis again to move back to UE coordinates
            yaw = util.reflect_radians_about_x_axis(yaw)
            transform = carla.Transform(carla.Location(x=x, y=y),carla.Rotation(yaw=yaw))
            trajectory.append(transform)

        if self.plot_scenario:
            """Plot scenario"""
            filename = f"agent{self.__ego_vehicle.id}_frame{frame}_lcss_control"
            ctrl_result.headings = headings
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
            ego_bbox = [lon, lat]
            params.update(self.__params)
            plot_lcss_prediction(pred_result, ovehicles, params, ctrl_result,
                    self.__prediction_horizon, ego_bbox, filename=filename)
        if self.plot_simulation:
            """Save planned trajectory for final plotting"""
            self.__plot_simulation_data.planned_trajectories[frame] = np.concatenate(
                    (params.x0[None], ctrl_result.X_star))
            self.__plot_simulation_data.planned_controls[frame] = ctrl_result.U_star
        return trajectory

    def do_first_step(self, frame):
        """Set the first frame and instantiate the online
        scene builder to collect environment data."""
        self.__first_frame = frame
        self.__scene_builder = self.__scene_builder_cls(
            self,
            self.__map_reader,
            self.__ego_vehicle,
            self.__other_vehicles,
            self.__lidar_feeds,
            "test",
            self.__first_frame,
            scene_config=self.__scene_config,
            debug=True)

    def run_step(self, frame, control=None):
        """Run motion planner step. Should be called whenever carla.World.click() is called.

        Parameters
        ==========
        frame : int
            Current frame of the simulation.
        control: carla.VehicleControl (optional)
            Optional control to apply to the motion planner. Used to move the vehicle
            while burning frames in the simulator before doing motion planning.
        """
        logging.debug(f"In LCSSHighLevelAgent.run_step() with frame = {frame}")
        if self.__first_frame is None:
            self.do_first_step(frame)
        
        self.__scene_builder.capture_trajectory(frame)
        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            """We only motion plan every `record_interval` frames
            (e.g. every 0.5 seconds of simulation)."""
            frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
            if frame_id < self.__n_burn_interval:
                """Initially collect data without doing any control to the vehicle."""
                pass
            elif (frame_id - self.__n_burn_interval) % self.__control_horizon == 0:
                trajectory = self.__compute_prediction_controls(frame)
                self.__local_planner.set_plan(trajectory, self.__scene_config.record_interval)
            if self.plot_simulation:
                """Save actual trajectory for final plotting"""
                payload = carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(self.__ego_vehicle, flip_y=True)
                self.__plot_simulation_data.actual_trajectory[frame] = payload

        if not control:
            control = self.__local_planner.run_step()
        self.__ego_vehicle.apply_control(control)

    def remove_scene_builder(self, first_frame):
        raise Exception(f"Can't remove scene builder from {util.classname(first_frame)}.")

    @staticmethod
    def parse_image(weak_self, image):
        """Pass sensor image to each scene builder.

        Parameters
        ==========
        image : carla.SemanticLidarMeasurement
        """
        self = weak_self()
        if not self:
            return
        logging.debug(f"in DataCollector.parse_image() player = {self.__ego_vehicle.id} frame = {image.frame}")
        self.__lidar_feeds[image.frame] = image
        if self.__scene_builder:
            self.__scene_builder.capture_lidar(image)
