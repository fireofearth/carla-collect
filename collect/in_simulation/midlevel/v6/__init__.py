"""
v6 is a modification of v2_2 that does original approach on curved road boundaries
and uses LTI bicycle model linearized around origin.

    - Still uses double integrator as model for ego vehicle.
    - Applies curved road boundaries using segmented polytopes.
"""

# Built-in libraries
import os
import logging
import collections
import weakref
import copy
import numbers
import math

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

try:
    from utils.trajectory_utils import prediction_output_to_trajectories
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from .util import plot_lcss_prediction, plot_oa_simulation
from ..util import (get_vertices_from_center, profile,
        get_approx_union, plot_h_polyhedron, get_ovehicle_color_set,
        plot_multiple_coinciding_controls, get_vertices_from_centers)
from ..dynamics import (get_state_matrix, get_input_matrix,
        get_output_matrix, get_feedforward_matrix)
from ..ovehicle import OVehicle
from ..prediction import generate_vehicle_latents
from ...lowlevel.v1_1 import LocalPlanner
from ....generate import AbstractDataCollector
from ....generate import create_semantic_lidar_blueprint
from ....generate.map import NaiveMapQuerier
from ....generate.scene import OnlineConfig, SceneBuilder
from ....generate.scene.v2_2.trajectron_scene import (
        TrajectronPlusPlusSceneBuilder)
from ....generate.scene.trajectron_util import (
        standardization, plot_trajectron_scene)

# Local libraries
import carla
import utility as util
import carlautil
import carlautil.debug

class MidlevelAgent(AbstractDataCollector):

    Z_SENSOR_REL = 2.5

    def __create_segmentation_lidar_sensor(self):
        return self.__world.spawn_actor(
                create_semantic_lidar_blueprint(self.__world),
                carla.Transform(carla.Location(z=self.Z_SENSOR_REL)),
                attach_to=self.__ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid)

    def __make_global_params(self):
        """Get Global LCSS parameters used across all loops"""
        params = util.AttrDict()
        # Slack variable for solver
        # TODO: hardcoded
        params.M_big = 1000
        # Control variable for solver, setting max acceleration
        # TODO: hardcoded
        params.max_a = 2
        params.min_a = -7
        # Maximum steering angle
        physics_control = self.__ego_vehicle.get_physics_control()
        wheels = physics_control.wheels
        params.limit_delta = np.deg2rad(wheels[0].max_steer_angle)
        params.max_delta = 0.5*params.limit_delta
        # Since vehicle has a max steering rate (unknown, add a reasonable constraint).
        params.max_delta_chg = 0.16*params.limit_delta
        # TODO: what are the com values?
        # (0.60, 0.0, -0.25)
        # lon_com = physics_control.center_of_mass.x
        # lat_com = physics_control.center_of_mass.y
        # hi_com = physics_control.center_of_mass.z
        # longitudinal and lateral dimensions of car are normally 3.70 m, 1.79 m resp.
        params.bbox_lon, params.bbox_lat, _ = carlautil.actor_to_bbox_ndarray(
                self.__ego_vehicle)
        # Number of faces of obstacle sets
        params.L = 4
        # Minimum distance from vehicle to avoid collision. 
        params.diag = np.sqrt(params.bbox_lon**2 + params.bbox_lat**2) / 2.
        return params

    def __setup_rectangular_boundary_conditions(self):
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
        # __goal
        #   Goal destination the vehicle should navigates to.
        self.__goal = util.AttrDict(x=50, y=0, is_relative=True)

    def __setup_curved_road_segmented_boundary_conditions(
            self, turn_choices, max_distance):
        # __turn_choices : list of int
        #   List of choices of turns to make at intersections,
        #   starting with the first intersection to the last.
        self.__turn_choices = turn_choices
        # __max_distance : number
        #   Maximum distance from road
        self.__max_distance = max_distance
        # __road_segs : util.AttrDict
        #   Container of road segment properties.
        # __road_segs.spline : scipy.interpolate.CubicSpline
        #   The spline representing the path the vehicle should motion plan on.
        # __road_segs.polytopes : list of (ndarray, ndarray)
        #   List of polytopes in H-representation (A, b) where x is in polytope if Ax <= b.
        # __road_segs.distances : ndarray
        #   The distances along the spline to follow from nearest endpoint
        #   before encountering corresponding covering polytope in index.
        # __road_segs.positions : ndarray
        #   The 2D positions of center of the covering polytope in index.
        self.__road_segs = self.__map_reader.curved_road_segments_enclosure_from_actor(
                    self.__ego_vehicle, self.__max_distance, choices=self.__turn_choices,
                    flip_y=True)
        logging.info(f"max curvature of planned path is {self.__road_segs.max_k}; "
                     f"created {len(self.__road_segs.polytopes)} polytopes covering "
                     f"a distance of {np.round(self.__max_distance, 2)} m in total.")
        x, y = self.__road_segs.spline(self.__road_segs.distances[-1])
        # __goal
        #   Not used for motion planning when using this BC.
        self.__goal = util.AttrDict(x=x, y=y, is_relative=False)

    def __init__(self,
            ego_vehicle,
            map_reader,
            other_vehicle_ids,
            eval_stg,
            n_burn_interval=4,
            prediction_horizon=8,
            control_horizon=6,
            step_horizon=1,
            n_predictions=100,
            scene_builder_cls=TrajectronPlusPlusSceneBuilder,
            scene_config=OnlineConfig(),
            #########################
            # Controller type setting
            agent_type="oa",
            n_coincide=6,
            #######################
            # Logging and debugging
            log_cplex=False,
            log_agent=False,
            plot_scenario=False,
            plot_simulation=False,
            plot_boundary=False,
            plot_vertices=False,
            plot_overapprox=False,
            #######################
            # Planned path settings
            turn_choices=[],
            max_distance=100,
            #######################
            **kwargs):
        self.__ego_vehicle = ego_vehicle
        self.__map_reader = map_reader
        self.__eval_stg = eval_stg
        # __n_burn_interval : int
        #   Interval in prediction timesteps to skip prediction and control.
        self.__n_burn_interval = n_burn_interval
        # __control_horizon : int
        #   Number of predictions timesteps to optimize control over.
        self.__control_horizon = control_horizon
        # __prediction_horizon : int
        #   Number of predictions timesteps to predict other vehicles over.
        self.__prediction_horizon = prediction_horizon
        # __step_horizon : int
        #   Number of steps to take at each iteration of MPC.
        self.__step_horizon = step_horizon
        # __n_predictions : int
        #   Number of predictions to generate on each control step.
        self.__n_predictions = n_predictions
        self.__scene_builder_cls = scene_builder_cls
        self.__scene_config = scene_config

        if agent_type == "oa":
            assert control_horizon <= prediction_horizon
            # assert n_coincide <= control_horizon
            pass
        elif agent_type == "mcc":
            raise NotImplementedError(f"Agent type {agent_type} not implemented.")
        elif agent_type == "rmcc":
            raise NotImplementedError(f"Agent type {agent_type} not implemented.")
        else:
            raise ValueError(f"Agent type {agent_type} is not recognized.")
        self.__agent_type = agent_type

        # __n_coincide : int
        #   Number of steps in motion plan that coincide.
        #   Used for multiple coinciding control. 
        self.__n_coincide = n_coincide

        # __first_frame : int
        #   First frame in simulation. Used to find current timestep.
        self.__first_frame = None
        self.__scene_builder = None

        self.__world = self.__ego_vehicle.get_world()
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

        self.__setup_curved_road_segmented_boundary_conditions(
                turn_choices, max_distance)
        
        self.log_cplex       = log_cplex
        self.log_agent       = log_agent
        self.plot_scenario   = plot_scenario
        self.plot_simulation = plot_simulation
        self.plot_boundary   = plot_boundary
        self.plot_vertices   = plot_vertices
        self.plot_overapprox = plot_overapprox
        if self.plot_simulation:
            self.__plot_simulation_data = util.AttrDict(
                actual_trajectory=collections.OrderedDict(),
                planned_trajectories=collections.OrderedDict(),
                planned_controls=collections.OrderedDict(),
                goals=collections.OrderedDict()
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
        if self.__agent_type == "oa":
            filename = f"agent{self.__ego_vehicle.id}_oa_simulation"
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
            plot_oa_simulation(
                self.__scene_builder.get_scene(),
                self.__plot_simulation_data.actual_trajectory,
                self.__plot_simulation_data.planned_trajectories,
                self.__plot_simulation_data.planned_controls,
                self.__road_segs,
                np.array([lon, lat]),
                self.__step_horizon,
                self.__prediction_timestep,
                filename=filename,
                road_boundary_constraints=False,
            )
        else:
            raise NotImplementedError()

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.__sensor.destroy()
        self.__sensor = None
        if self.plot_simulation:
            self.__plot_simulation()

    def do_prediction(self, frame):
        """Get processed scene object from scene builder, input the scene to a model to
        generate the predictions, and then return the predictions and the latents variables."""

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

    def get_current_steering(self):
        """Get current steering angle in radians.
        TODO: is this correct??
        https://github.com/carla-simulator/carla/issues/699
        No it is not really correct. Should update to CARLA 0.9.12 and fix.
        """
        return -self.__ego_vehicle.get_control().steer*self.__params.limit_delta
    
    def get_current_velocity(self):
        """
        Get current velocity of vehicle in
        """
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(
                self.__ego_vehicle, flip_y=True)
        return np.sqrt(v_0_x**2 + v_0_y**2)

    def make_local_params(self, frame, ovehicles):
        """Get the linearized bicycle model using the vehicle's
        immediate orientation."""
        params = util.AttrDict()
        params.frame = frame

        """Dynamics parameters"""
        p_0_x, p_0_y, _ = carlautil.to_location_ndarray(
                self.__ego_vehicle, flip_y=True)
        
        _, psi_0, _ = carlautil.actor_to_rotation_ndarray(
                self.__ego_vehicle, flip_y=True)
        v_0_mag = self.get_current_velocity()
        delta_0 = self.get_current_steering()
        # initial_state - state at current frame in world/local coordinates
        #   Local coordinates has initial position and heading at 0
        initial_state = util.AttrDict(
            world=np.array([p_0_x, p_0_y, psi_0, v_0_mag, delta_0]),
            local=np.array([0,     0,     0,     v_0_mag, delta_0])
        )
        params.initial_state = initial_state
        # transform - transform points from local coordinates back to world coordinates.
        M = np.array([
            [math.cos(psi_0), -math.sin(psi_0), p_0_x],
            [math.sin(psi_0),  math.cos(psi_0), p_0_y],
        ])
        def transform(X):
            points = np.pad(X[:, :2], [(0, 0), (0, 1)], mode="constant", constant_values=1)
            points = util.obj_matmul(points, M.T)
            psis   = X[:, 2] + psi_0
            return np.concatenate((points, psis[..., None], X[:, 3:]), axis=1)
        params.transform = transform
        # longitudinal and lateral dimensions of car are normally 3.70 m, 1.79 m resp.
        bbox_lon = self.__params.bbox_lon
        # TODO: use center of gravity from API instead
        A = get_state_matrix(0, 0, 0, v_0_mag, delta_0,
                l_r=0.5*bbox_lon, L=bbox_lon)
        B = get_input_matrix()
        C = get_output_matrix()
        D = get_feedforward_matrix()
        sys = control.matlab.c2d(control.matlab.ss(A, B, C, D),
                self.__prediction_timestep)
        A = np.array(sys.A)
        B = np.array(sys.B)
        # nx, nu - size of state variable and control input respectively.
        nx, nu = sys.B.shape
        params.nx, params.nu = nx, nu
        T = self.__control_horizon
        # Closed form solution to states given inputs
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
        Gamma = np.concatenate((np.zeros((nx, T*nu,)), np.linalg.solve(Abar, Bbar),))
        params.Gamma = Gamma
        # make state computation account for initial position and velocity
        initial_local_rollout = np.concatenate(
                [np.linalg.matrix_power(A, t) @ initial_state.local for t in range(T+1)])
        params.initial_rollout = util.AttrDict(local=initial_local_rollout)

        """Other vehicles"""
        # O - number of obstacles
        params.O = len(ovehicles)
        # K - for each o=1,...,O K[o] is the number of outer approximations for vehicle o
        params.K = np.zeros(params.O, dtype=int)
        for idx, vehicle in enumerate(ovehicles):
            params.K[idx] = vehicle.n_states

        """Local params for (random) multiple coinciding controls."""
        # N_traj - number of planned trajectories possible to compute
        params.N_traj = np.prod(params.K)
        max_K = np.max(params.K)
        # TODO: figure out a way to specify number of subtrajectories
        params.N_select = min(int(1.5*max_K), params.N_traj)
        # subtraj_indices - the indices of the sub-trajectory to optimize for
        params.subtraj_indices = np.random.choice(
                np.arange(params.N_traj), size=params.N_select, replace=False)

        return params

    def __plot_segs_polytopes(self, params, segs_polytopes, goal):
        fig, ax = plt.subplots(figsize=(7, 7))
        x_min, y_min = np.min(self.__road_segs.positions, axis=0)
        x_max, y_max = np.max(self.__road_segs.positions, axis=0)
        self.__map_reader.render_map(ax,
                extent=(x_min - 20, x_max + 20, y_min - 20, y_max + 20))
        x, y, _ = carlautil.to_location_ndarray(self.__ego_vehicle, flip_y=True)
        ax.scatter(x, y, c="r", zorder=10)
        x, y = goal
        ax.scatter(x, y, c="g", marker="*", zorder=10)
        for A, b in segs_polytopes:
            util.npu.plot_h_polyhedron(ax, A, b, fc='b', ec='b', alpha=0.3)
        filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_boundary"
        fig.savefig(os.path.join('out', f"{filename}.png"))
        fig.clf()

    def compute_segs_polytopes_and_goal(self, params):
        """
        TODO: uses the ego vehicle's current speed limit to compute how
        many polytopes to use as constraints. what happens when speed
        limit changes?
        TODO: speed limit is not enough to infer distance to motion plan.
        Also need curvature since car slows down on road.
        """
        n_segs = len(self.__road_segs.polytopes)
        segment_length = self.__road_segs.segment_length
        v_lim = self.__ego_vehicle.get_speed_limit()
        go_forward = int(
            (0.75*v_lim*self.__prediction_timestep*self.__prediction_horizon) \
                // segment_length + 1
        )
        pos0 = params.initial_state.world[:2]
        closest_idx = np.argmin(
                np.linalg.norm(self.__road_segs.positions - pos0, axis=1))
        near_idx = max(closest_idx - 1, 0)
        far_idx = min(closest_idx + go_forward, n_segs)
        segs_polytopes = self.__road_segs.polytopes[near_idx:far_idx]
        goal_idx = min(closest_idx + go_forward, n_segs - 1)
        goal = self.__road_segs.positions[goal_idx]

        psi_0 = params.initial_state.world[2]
        seg_psi_bounds = []
        epsilon = (1/6)*np.pi
        for (x_1, y_1), (x_2, y_2) in self.__road_segs.tangents[near_idx:far_idx]:
            theta_1 = util.npu.warp_radians_0_to_2pi(math.atan2(y_1, x_1)) - psi_0
            theta_2 = util.npu.warp_radians_0_to_2pi(math.atan2(y_2, x_2)) - psi_0
            theta_1 = util.npu.warp_radians_neg_pi_to_pi(theta_1)
            theta_2 = util.npu.warp_radians_neg_pi_to_pi(theta_2)
            seg_psi_bounds.append(
                (min(theta_1, theta_2) - epsilon, max(theta_1, theta_2) + epsilon)
            )

        if self.plot_boundary:
            self.__plot_segs_polytopes(params, segs_polytopes, goal)
        return segs_polytopes, goal, seg_psi_bounds
    
    def compute_dyamics_constraints(self, params, v, delta):
        """Set velocity magnitude constraints.
        Usually street speed limits are 30 km/h == 8.33.. m/s.
        Speed limits can be 30, 40, 60, 90 km/h

        Parameters
        ==========
        v : np.array of docplex.mp.vartype.VarType
        """
        max_v = self.__ego_vehicle.get_speed_limit() # is m/s
        max_delta = self.__params.max_delta
        constraints = []
        constraints.extend([ z <= max_v      for z in v ])
        constraints.extend([ z >= 0          for z in v ])
        constraints.extend([z  <=  max_delta for z in delta])
        constraints.extend([z  >= -max_delta for z in delta])
        return constraints

    def __compute_vertices(self, params, ovehicles):
        """Compute verticles from predictions."""
        K, n_ov = params.K, params.O
        T = self.__prediction_horizon
        vertices = np.empty((T, np.max(K), n_ov), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    ps = ovehicle.pred_positions[latent_idx][:,t]
                    yaws = ovehicle.pred_yaws[latent_idx][:,t]
                    vertices[t][latent_idx][ov_idx] = get_vertices_from_centers(
                            ps, yaws, ovehicle.bbox)

        return vertices

    def __plot_overapproximations(self, params, ovehicles, vertices, A_union, b_union):
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
                for t in range(self.__prediction_horizon):
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
        filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_overapprox"
        fig.savefig(os.path.join('out', f"{filename}.png"))
        fig.clf()

    def __compute_overapproximations(self, vertices, params, ovehicles):
        """Compute the approximation of the union of obstacle sets.

        Parameters
        ==========
        vertices : ndarray
            Verticles to for overapproximations.
        params : util.AttrDict
            Parameters of motion planning problem.
        ovehicles : list of OVehicle
            Vehicles to compute overapproximations from.

        Returns
        =======
        ndarray
            Case 1 agent type is oa:
                Collection of A matrices of shape (T, max(K), O, L, 2).
                Where max(K) largest set of cluster.
            Case 2 agent type is (r)mcc:
                Collection of A matrices of shape (N_traj, T, O, L, 2).
                Axis 2 (zero-based) is sorted by ovehicle.
        ndarray
            Case 1 agent type is oa:
                Collection of b vectors of shape (T, max(K), O, L).
                Where max(K) largest set of cluster.
            Case 2 agent type is (r)mcc:
                Collection of b vectors of shape (N_traj, T, O, L).
                Axis 2 (zero-based) is sorted by ovehicle.
        """
        if self.__agent_type == "oa":
            K, n_ov = params.K, params.O
            T = self.__prediction_horizon
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
                self.__plot_overapproximations(params, ovehicles, vertices, A_union, b_union)

            return A_union, b_union
        elif self.__agent_type == "mcc" or self.__agent_type == "rmcc":
            N_traj, O = params.N_traj, params.O
            T = self.__prediction_horizon
            A_unions = np.empty((N_traj, T, O,), dtype=object).tolist()
            b_unions = np.empty((N_traj, T, O,), dtype=object).tolist()
            for traj_idx, latent_indices in enumerate(
                    util.product_list_of_list([range(ovehicle.n_states) for ovehicle in ovehicles])):
                for t in range(T):
                    for ov_idx, ovehicle in enumerate(ovehicles):
                        latent_idx = latent_indices[ov_idx]
                        yaws = ovehicle.pred_yaws[latent_idx][:,t]
                        vertices_k = vertices[t][latent_idx][ov_idx]
                        mean_theta_k = np.mean(yaws)
                        A_union_k, b_union_k = get_approx_union(mean_theta_k, vertices_k)
                        A_unions[traj_idx][t][ov_idx] = A_union_k
                        b_unions[traj_idx][t][ov_idx] = b_union_k

            return np.array(A_unions), np.array(b_unions)
        else:
            raise NotImplementedError()
    
    def do_multiple_control(self, params, ovehicles):
        """Decide parameters.

        Since we're doing multiple coinciding, we want to compute...

        TODO finish description 
        TODO update
        """
        vertices = self.__compute_vertices(params, ovehicles)
        A_unions, b_unions = self.__compute_overapproximations(vertices, params, ovehicles)
        
        """Apply motion planning problem"""
        # params.N_select params.N_traj params.subtraj_indices
        N_select, L, K, O, Gamma, nu, nx = params.N_traj, self.__params.L, \
                params.K, params.O, params.Gamma, params.nu, params.nx
        T = self.__control_horizon
        max_a = self.__params.max_a
        model = docplex.mp.model.Model(name='obstacle_avoidance')
        # set up controls variables for each trajectory
        u = np.array(model.continuous_var_list(N_select*T*nu, lb=-max_a, ub=max_a, name='control'))
        Delta = np.array(model.binary_var_list(O*L*N_select*T, name='delta')).reshape(N_select, T, O, L)

        raise NotImplementedError()

    def do_highlevel_control(self, params, ovehicles):
        """Decide parameters.

        TODO finish description 
        """
        vertices = self.__compute_vertices(params, ovehicles)
        A_union, b_union = self.__compute_overapproximations(vertices, params, ovehicles)
        segs_polytopes, goal, seg_psi_bounds = self.compute_segs_polytopes_and_goal(params)

        """Apply motion planning problem"""
        n_segs = len(segs_polytopes)
        L, K, Gamma, nu, nx = self.__params.L, params.K, \
                params.Gamma, params.nu, params.nx
        T = self.__control_horizon
        max_a, min_a, max_delta_chg = self.__params.max_a, self.__params.min_a, self.__params.max_delta_chg
        model = docplex.mp.model.Model(name="proposed_problem")
        min_u = np.vstack((np.full(T, min_a), np.full(T, -max_delta_chg))).T.ravel()
        max_u = np.vstack((np.full(T, max_a), np.full(T,  max_delta_chg))).T.ravel()
        u = np.array(model.continuous_var_list(nu*T, lb=min_u, ub=max_u, name='u'),
                dtype=object)
        # Slack variables for vehicle obstacles
        Delta = np.array(model.binary_var_list(L*np.sum(K)*T, name='delta'),
                dtype=object).reshape(np.sum(K), T, L)
        # Slack variables from road obstacles
        Omicron = np.array(model.binary_var_list(n_segs*T, name="omicron"),
                dtype=object).reshape(n_segs, T)
        # State variables
        X = (params.initial_rollout.local + util.obj_matmul(Gamma, u)).reshape(T + 1, nx)
        X = params.transform(X)
        X = X[1:]
        # Control variables
        U = u.reshape(T, nu)

        """Apply motion dynamics constraints"""
        model.add_constraints(self.compute_dyamics_constraints(params, X[:, 3], X[:, 4]))

        """Apply road boundary constraints"""
        M_big, diag = self.__params.M_big, self.__params.diag
        psi_0 = params.initial_state.world[2]
        for t in range(self.__control_horizon):
            for seg_idx, (A, b) in enumerate(segs_polytopes):
                lhs = util.obj_matmul(A, X[t, :2]) - np.array(M_big*(1 - Omicron[seg_idx, t]))
                rhs = b# - diag
                """Constraints on road boundaries"""
                model.add_constraints(
                        [l <= r for (l,r) in zip(lhs, rhs)])
                """Constraints on angle boundaries"""
                # disabling do to failure in some circumstances.
                # lhs = X[t, 2] - psi_0 - M_big*(1 - Omicron[seg_idx, t])
                # model.add_constraint(lhs <= seg_psi_bounds[seg_idx][1])
                # lhs = X[t, 2] - psi_0 + M_big*(1 - Omicron[seg_idx, t])
                # model.add_constraint(lhs >= seg_psi_bounds[seg_idx][0])
            model.add_constraint(np.sum(Omicron[:, t]) >= 1)

        """Apply vehicle collision constraints"""
        K, diag, M_big = params.K, \
                self.__params.diag, self.__params.M_big
        for ov_idx, ovehicle in enumerate(ovehicles):
            n_states = ovehicle.n_states
            sum_clu = np.sum(K[:ov_idx])
            for latent_idx in range(n_states):
                for t in range(self.__control_horizon):
                    A_obs = A_union[t][latent_idx][ov_idx]
                    b_obs = b_union[t][latent_idx][ov_idx]
                    indices = sum_clu + latent_idx
                    lhs = util.obj_matmul(A_obs, X[t, :2]) + M_big*(1 - Delta[indices, t])
                    rhs = b_obs + diag
                    model.add_constraints([l >= r for (l,r) in zip(lhs, rhs)])
                    model.add_constraint(np.sum(Delta[indices, t]) >= 1)
        
        """Start from current vehicle position and minimize the objective"""
        w_ch_accel = 0.
        w_ch_turning = 0.
        w_accel = 0.
        w_turning = 0.
        # final destination objective
        cost = (X[-1, 0] - goal[0])**2 + (X[-1, 1] - goal[1])**2
        # change in acceleration objective
        for u1, u2 in util.pairwise(U[:, 0]):
            _u = (u1 - u2)
            cost += w_ch_accel*_u*_u
        # change in turning rate objective
        for u1, u2 in util.pairwise(U[:, 1]):
            _u = (u1 - u2)
            cost += w_ch_turning*_u*_u
        cost += w_accel*np.sum(U[:, 0]**2)
        # turning rate objective
        cost += w_turning*np.sum(U[:, 1]**2)
        model.minimize(cost)
        # TODO: warmstarting
        # if self.U_warmstarting is not None:
        #     # Warm start inputs if past iteration was run.
        #     warm_start = model.new_solution()
        #     for i, u in enumerate(self.U_warmstarting[self.__control_horizon:]):
        #         warm_start.add_var_value(f"u_{2*i}", u[0])
        #         warm_start.add_var_value(f"u_{2*i + 1}", u[1])
        #     # add delta_0 as hotfix to MIP warmstart as it needs
        #     # at least 1 integer value set.
        #     warm_start.add_var_value('delta_0', 0)
        #     model.add_mip_start(warm_start)
        
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
                goal=goal)

    def __plot_scenario(self, pred_result, ovehicles, params, ctrl_result):
        if self.__agent_type == "oa":
            filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_lcss_control"
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
            ego_bbox = np.array([lon, lat])
            params.update(self.__params)
            plot_lcss_prediction(pred_result, ovehicles, params, ctrl_result,
                    self.__control_horizon, ego_bbox, filename=filename)

    # @profile(sort_by='cumulative', lines_to_print=50, strip_dirs=True)
    def __compute_prediction_controls(self, frame):
        pred_result = self.do_prediction(frame)
        ovehicles = self.make_ovehicles(pred_result)
        params = self.make_local_params(frame, ovehicles)
        ctrl_result = self.do_highlevel_control(params, ovehicles)

        """use control input next round for warm starting."""
        # self.U_warmstarting = ctrl_result.U_star

        """Get trajectory and velocity"""
        trajectory = []
        velocity = []
        X = np.concatenate((params.initial_state.world[None], ctrl_result.X_star))
        n_steps = X.shape[0]
        for t in range(1, n_steps):
            x, y, yaw = X[t, :3]
            y = -y # flip about x-axis again to move back to UE coordinates
            # flip about x-axis again to move back to UE coordinates
            yaw = np.rad2deg(util.reflect_radians_about_x_axis(yaw))
            transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation(yaw=yaw))
            trajectory.append(transform)
            velocity.append(X[t, 3])

        if self.plot_scenario:
            """Plot scenario"""
            self.__plot_scenario(pred_result, ovehicles, params, ctrl_result)
        if self.plot_simulation:
            """Save planned trajectory for final plotting"""
            if self.__agent_type == "oa":
                self.__plot_simulation_data.planned_trajectories[frame] = X
                self.__plot_simulation_data.planned_controls[frame] = ctrl_result.U_star
                self.__plot_simulation_data.goals[frame] = ctrl_result.goal
                self.__plot_simulation_data
            else:
                raise NotImplementedError()
        return trajectory, velocity

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
            debug=False)

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
            elif (frame_id - self.__n_burn_interval) % self.__step_horizon == 0:
                trajectory, velocity = self.__compute_prediction_controls(frame)
                self.__local_planner.set_plan(trajectory,
                        self.__scene_config.record_interval, velocity=velocity)
            if self.plot_simulation:
                """Save actual trajectory for final plotting"""
                payload = carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(self.__ego_vehicle, flip_y=True)
                payload = np.array([
                        payload[0], payload[1], payload[13],
                        self.get_current_velocity(),
                        self.get_current_steering()])
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
