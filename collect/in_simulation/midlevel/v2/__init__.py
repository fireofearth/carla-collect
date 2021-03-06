"""
v2 is a generalization of v1.

    - This is the original approach motion planner.
    - It clusters predictions from other vehicles and computes a trajectory using a double integrator model for the ego vehicle.
    - The dynamics are adapted based on the vehicle's heading at each iteration of the MPC.
"""

import sys
import os
import math
import logging
import collections
import weakref
import copy

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
import carla

import utility as util
import carlautil
import carlautil.debug

try:
    from utils.trajectory_utils import prediction_output_to_trajectories
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from ..util import (get_vertices_from_center, obj_matmul, profile,
        get_approx_union, plot_h_polyhedron, get_ovehicle_color_set,
        plot_lcss_prediction)
from ..ovehicle import OVehicle
from ..prediction import generate_vehicle_latents
from ...lowlevel import LocalPlanner
from ....generate import AbstractDataCollector
from ....generate import create_semantic_lidar_blueprint
from ....generate.map import NaiveMapQuerier
from ....generate.scene import OnlineConfig, SceneBuilder
from ....generate.scene.v2_1.trajectron_scene import (
        TrajectronPlusPlusSceneBuilder)
from ....generate.scene.v2_1.trajectron_scene import (
        standardization, print_and_reset_specs, plot_trajectron_scene)

class MidlevelAgent(AbstractDataCollector):
    """Controller for vehicle using predictions."""

    Z_SENSOR_REL = 2.5

    def __create_segmentation_lidar_sensor(self):
        return self.__world.spawn_actor(
                create_semantic_lidar_blueprint(self.__world),
                carla.Transform(carla.Location(z=self.Z_SENSOR_REL)),
                attach_to=self.__ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid)

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
            scene_config=OnlineConfig()):
        
        self.__ego_vehicle = ego_vehicle
        self.__map_reader = map_reader
        self.__world = self.__ego_vehicle.get_world()

        # __first_frame : int
        #     First frame in simulation. Used to find current timestep.
        self.__first_frame = None
        self.__scene_builder = None
        self.__scene_config = scene_config
        self.__scene_builder_cls = scene_builder_cls
        self.__control_horizon = control_horizon
        self.__n_burn_interval = n_burn_interval
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

        self.__local_planner = LocalPlanner(self.__ego_vehicle)

        self.__goal = util.AttrDict(x=50, y=0, is_relative=True)
    
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

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.__sensor.destroy()
        self.__sensor = None

    def do_prediction(self, frame):

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

    @staticmethod
    def get_state_space_representation(pred_result):
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
        sys = control.matlab.c2d(control.matlab.ss(A, B, C, D), pred_result.scene.dt)
        A = np.array(sys.A)
        B = np.array(sys.B)
        return (A, B)

    def make_highlevel_params(self, pred_result, ovehicles):
        """Get LCSS parameters"""
        # TODO: refactor this long chain

        params = util.AttrDict()
        params.A, params.B = self.get_state_space_representation(pred_result)
        # number of state variables x, number of input variables u
        # nx = 4, nu = 2
        params.nx, params.nu = params.B.shape
        bbox_lon, bbox_lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
        params.diag = np.sqrt(bbox_lon**2 + bbox_lat**2) / 2.
        # Prediction parameters
        params.T = self.__prediction_horizon
        params.O = len(ovehicles) # number of obstacles
        params.L = 4 # number of faces of obstacle sets
        params.K = np.zeros(params.O, dtype=int)
        for idx, vehicle in enumerate(ovehicles):
            params.K[idx] = vehicle.n_states

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

        p_0_x, p_0_y, _ = carlautil.actor_to_location_ndarray(self.__ego_vehicle)
        p_0_y = -p_0_y # need to flip about x-axis
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(self.__ego_vehicle)
        v_0_y = -v_0_y # need to flip about x-axis
        # x0 : np.array
        #   Initial state
        x0 = np.array([p_0_x, p_0_y, v_0_x, v_0_y])
        A, T = params.A, params.T
        # States_free_init has shape (nx*(T+1))
        params.States_free_init = np.concatenate([
                np.linalg.matrix_power(A, t) @ x0 for t in range(T+1)])
        return params

    def compute_velocity_constraints(self, v_x, v_y):
        """Velocity (3rd and 4th states) have coupled constraints.
        Generate docplex constraints for velocity for double integrators.

        Street speed limit is 30 km/h == 8.33.. m/s

        Parameters
        ==========
        v_x : np.array of docplex.mp.vartype.VarType
        v_y : np.array of docplex.mp.vartype.VarType
        """
        v_lim = self.__ego_vehicle.get_speed_limit() # is m/s
        _, theta, _ = carlautil.actor_to_rotation_ndarray(self.__ego_vehicle)
        theta = util.reflect_radians_about_x_axis(theta)
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
        """
        Present performance cars are capable of going from 0 to 60 mph in under 5 seconds.

        Reference:
        https://en.wikipedia.org/wiki/0_to_60_mph

        Parameters
        ==========
        u_x : np.array of docplex.mp.vartype.VarType
        u_y : np.array of docplex.mp.vartype.VarType
        """
        _, theta, _ = carlautil.actor_to_rotation_ndarray(self.__ego_vehicle)
        theta = util.reflect_radians_about_x_axis(theta)
        r = -2.5
        a_1 = 7.5
        a_2 = 5.0
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

    def do_highlevel_control(self, params, ovehicles):
        """Decide parameters"""
        # TODO: assign eps^k_o and beta^k_o to the vehicles.
        # Skipping that for now.
    
        """Compute the approximation of the union of obstacle sets"""
        # Find vertices of sampled obstacle sets
        T, K, n_ov = params.T, params.K, params.O
        vertices = np.empty((T, np.max(K), n_ov), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    ps = ovehicle.pred_positions[latent_idx]
                    yaws = ovehicle.pred_yaws[latent_idx]
                    n_p = ps.shape[0]
                    vertices[t][latent_idx][ov_idx] = np.zeros((n_p, 8))
                    for k in range(n_p):
                        vertices[t][latent_idx][ov_idx][k] = get_vertices_from_center(
                                ps[k,t], yaws[k,t], ovehicle.bbox)
        
        plot_vertices = False
        if plot_vertices:
            t = self.__prediction_horizon - 1
            latent_idx = 0
            ov_idx = 0
            # for ovehicle in scene.ovehicles:
            X = vertices[t][latent_idx][ov_idx][:,0:2].T
            plt.scatter(X[0], X[1], c='r', s=2)
            X = vertices[t][latent_idx][ov_idx][:,2:4].T
            plt.scatter(X[0], X[1], c='b', s=2)
            X = vertices[t][latent_idx][ov_idx][:,4:6].T
            plt.scatter(X[0], X[1], c='g', s=2)
            X = vertices[t][latent_idx][ov_idx][:,6:8].T
            plt.scatter(X[0], X[1], c='orange', s=2)
            plt.gca().set_aspect('equal')
            plt.show()

        """Cluster the samples"""
        T, K, n_ov = params.T, params.K, params.O
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
        plot_overapprox = False
        if plot_overapprox:
            fig, ax = plt.subplots()
            t = 7
            ovehicle = ovehicles[1]
            ps = ovehicle.pred_positions[0][:,t].T
            ax.scatter(ps[0], ps[1], c='r', s=2)
            ps = ovehicle.pred_positions[1][:,t].T
            ax.scatter(ps[0], ps[1], c='g', s=2)
            ps = ovehicle.pred_positions[2][:,t].T
            ax.scatter(ps[0], ps[1], c='b', s=2)
            plot_h_polyhedron(ax, A_union[t][0][1], b_union[t][0][1], fc='r', ec='k', alpha=0.3)
            plot_h_polyhedron(ax, A_union[t][1][1], b_union[t][1][1], fc='g', ec='k', alpha=0.3)
            plot_h_polyhedron(ax, A_union[t][2][1], b_union[t][2][1], fc='b', ec='k', alpha=0.3)
            ax.set_aspect('equal')
            plt.show()
        
        """Apply motion planning problem"""
        model = docplex.mp.model.Model(name="proposed_problem")
        L, T, K, Gamma, nu, nx = params.L, params.T, params.K, params.Gamma, params.nu, params.nx
        u = np.array(model.continuous_var_list(nu*T, lb=-np.inf, name='u'), dtype=object)
        delta_tmp = model.binary_var_matrix(L*np.sum(K), T, name='delta')
        delta = np.empty((L*np.sum(K), T,), dtype=object)
        for k, v in delta_tmp.items():
            delta[k] = v
        
        x_future = params.States_free_init + obj_matmul(Gamma, u)
        # TODO: hardcoded value, need to specify better
        big_M = 1000 # conservative upper-bound

        x1 = x_future[nx::nx]
        x2 = x_future[nx + 1::nx]
        x3 = x_future[nx + 2::nx]
        x4 = x_future[nx + 3::nx]
        u1 = u[0::nu]
        u2 = u[1::nu]

        model.add_constraints(self.compute_velocity_constraints(x3, x4))
        model.add_constraints(self.compute_acceleration_constraints(u1, u2))

        X = np.stack([x1, x2], axis=1)
        T, K, diag = params.T, params.K, params.diag
        for ov_idx, ovehicle in enumerate(ovehicles):
            n_states = ovehicle.n_states
            sum_clu = np.sum(K[:ov_idx])
            for latent_idx in range(n_states):
                for t in range(T):
                    A_obs = A_union[t][latent_idx][ov_idx]
                    b_obs = b_union[t][latent_idx][ov_idx]
                    indices = 4*(sum_clu + latent_idx) + np.arange(0,4)
                    lhs = obj_matmul(A_obs, X[t]) + big_M*(1-delta[indices,t])
                    rhs = b_obs + diag
                    model.add_constraints([l >= r for (l,r) in zip(lhs, rhs)])
                    model.add_constraint(np.sum(delta[indices, t]) >= 1)
        
        # start from current vehicle position and minimize the objective
        p_x, p_y, _ = carlautil.actor_to_location_ndarray(
                self.__ego_vehicle)
        p_y = -p_y # need to flip about x-axis
        if self.__goal.is_relative:
            goal_x, goal_y = p_x + self.__goal.x, p_y + self.__goal.y
        else:
            goal_x, goal_y = self.__goal.x, self.__goal.y
        start = np.array([p_x, p_y])
        goal = np.array([goal_x, goal_y])
        cost = (x1[-1] - goal_x)**2 + (x2[-1] - goal_y)**2
        model.minimize(cost)
        # model.print_information()
        model.parameters.mip.display = 5
        model.parameters.read.datacheck = 1
        s = model.solve(log_output=True)
        # model.print_solution()

        u_star = np.array([ui.solution_value for ui in u])
        cost = cost.solution_value
        x1 = np.array([x1i.solution_value for x1i in x1])
        x2 = np.array([x2i.solution_value for x2i in x2])
        X_star = np.stack((x1, x2)).T
        return util.AttrDict(cost=cost, u_star=u_star, X_star=X_star,
                A_union=A_union, b_union=b_union, vertices=vertices,
                start=start, goal=goal)

    @profile(sort_by='cumulative', lines_to_print=50, strip_dirs=True)
    def __compute_prediction_controls(self, frame):
        pred_result = self.do_prediction(frame)
        ovehicles = self.make_ovehicles(pred_result)
        params = self.make_highlevel_params(pred_result, ovehicles)
        ctrl_result = self.do_highlevel_control(params, ovehicles)

        """Get trajectory"""
        trajectory = []
        x, y, _ = carlautil.actor_to_location_ndarray(
                self.__ego_vehicle)
        X = np.concatenate((np.array([x, y])[None], ctrl_result.X_star))
        n_steps = X.shape[0]
        headings = []
        for t in range(1, n_steps):
            x, y = X[t]
            y = -y
            yaw = np.arctan2(X[t, 1] - X[t - 1, 1], X[t, 0] - X[t - 1, 0])
            headings.append(yaw)
            yaw = util.reflect_radians_about_x_axis(yaw)
            transform = carla.Transform(
                    carla.Location(x=x, y=y),
                    carla.Rotation(yaw=yaw))
            trajectory.append(transform)

        """Plot scenario"""
        plot_scenario = False
        if plot_scenario:
            ctrl_result.headings = headings
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
            ego_bbox = [lon, lat]
            plot_lcss_prediction(pred_result, ovehicles, params, ctrl_result,
                    self.__prediction_horizon, ego_bbox)

        return trajectory

    def do_first_step(self, frame):
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
        """
        Parameters
        ==========
        frame : int
        control: carla.VehicleControl (optional)
        """
        logging.debug(f"In LCSSHighLevelAgent.run_step() with frame = {frame}")
        if self.__first_frame is None:
            self.do_first_step(frame)
        
        self.__scene_builder.capture_trajectory(frame)
        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
            """Initially collect data without doing anything to the vehicle."""
            if frame_id < self.__n_burn_interval:
                pass
            elif (frame_id - self.__n_burn_interval) % self.__control_horizon == 0:
                trajectory = self.__compute_prediction_controls(frame)
                self.__local_planner.set_plan(trajectory, self.__scene_config.record_interval)

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
