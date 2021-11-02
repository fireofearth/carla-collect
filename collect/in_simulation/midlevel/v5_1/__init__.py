"""
v5 modifies v2_2 (original approach), v3 (MCC), v4 (RMCC) for curved boundaries.

    - Still uses double integrator as model for ego vehicle.
    - Applies curved road boundaries using segmented polytopes.

MCC, RMCC are not finished.
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
        plot_multiple_coinciding_controls, get_vertices_from_centers,
        plot_lcss_prediction, plot_oa_simulation, plot_multiple_simulation)
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

class MidlevelAgent(AbstractDataCollector):

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
        # longitudinal and lateral dimensions of car are normally 3.70 m, 1.79 m resp.
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
            n_burn_interval=4,
            control_horizon=6,
            prediction_horizon=8,
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
        #   Number of predictions timesteps to conduct control over.
        self.__control_horizon = control_horizon
        # __prediction_horizon : int
        #   Number of predictions timesteps to predict other vehicles over.
        self.__prediction_horizon = prediction_horizon
        # __n_predictions : int
        #   Number of predictions to generate on each control step.
        self.__n_predictions = n_predictions
        self.__scene_builder_cls = scene_builder_cls
        self.__scene_config = scene_config

        if agent_type == "oa":
            assert control_horizon <= prediction_horizon
        elif agent_type == "mcc" or agent_type == "rmcc":
            assert n_coincide <= prediction_horizon
            assert control_horizon <= n_coincide
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

        ####################################################
        # rectangular boundary constraints are not used here
        ####################################################

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

        ############################################
        # curved road segmented boundary constraints
        ############################################

        # __turn_choices : list of int
        #   List of choices of turns to make at intersections,
        #   starting with the first intersection to the last.
        self.__turn_choices = turn_choices
        # __max_distance : number
        #   Maximum distance from road
        self.__max_distance = max_distance
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
        #   Goal destination the vehicle should navigates to.
        #   Not used for motion planning.
        self.__goal = util.AttrDict(x=x, y=y, is_relative=False)
        
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
        if self.__agent_type == "oa":
            filename = f"agent{self.__ego_vehicle.id}_oa_simulation"
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
            plot_oa_simulation(
                self.__scene_builder.get_scene(),
                self.__plot_simulation_data.actual_trajectory,
                self.__plot_simulation_data.planned_trajectories,
                self.__plot_simulation_data.planned_controls,
                self.__road_segs,
                [lon, lat],
                self.__control_horizon,
                filename=filename
            )
        elif self.__agent_type == "mcc":
            filename = f"agent{self.__ego_vehicle.id}_mcc_simulation"
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
            plot_multiple_simulation(
                self.__scene_builder.get_scene(),
                self.__plot_simulation_data.actual_trajectory,
                self.__plot_simulation_data.planned_trajectories,
                self.__plot_simulation_data.planned_controls,
                self.__road_segs,
                [lon, lat],
                self.__control_horizon,
                filename=filename
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

    def make_local_params(self, frame, ovehicles):
        """Get Local LCSS parameters that are environment dependent."""
        params = util.AttrDict()

        """General local params."""
        params.frame = frame
        # O - number of obstacles
        params.O = len(ovehicles)
        # K - for each o=1,...,O K[o] is the number of outer approximations for vehicle o
        params.K = np.zeros(params.O, dtype=int)
        for idx, vehicle in enumerate(ovehicles):
            params.K[idx] = vehicle.n_states
        p_0_x, p_0_y, _ = carlautil.to_location_ndarray(
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
            (0.75 * v_lim*self.__prediction_timestep*self.__prediction_horizon) \
                // segment_length + 1
        )
        pos0 = params.x0[:2]
        closest_idx = np.argmin(np.linalg.norm(self.__road_segs.positions - pos0, axis=1))
        near_idx = max(closest_idx - 1, 0)
        far_idx = min(closest_idx + go_forward, n_segs)
        segs_polytopes = self.__road_segs.polytopes[near_idx:far_idx]
        goal_idx = min(closest_idx + go_forward, n_segs - 1)
        goal = self.__road_segs.positions[goal_idx]

        if self.plot_boundary:
            self.__plot_segs_polytopes(params, segs_polytopes, goal)
        return segs_polytopes, goal
    
    def compute_velocity_constraints(self, v_x, v_y):
        """Velocity states have coupled constraints.
        Generate docplex constraints for velocity for double integrators.

        Note often street speed limit is 30 km/h == 8.33.. m/s

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

        return vertices

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
            T, K, O = self.__params.T, params.K, params.O
            A_union = np.empty((T, np.max(K), O,), dtype=object).tolist()
            b_union = np.empty((T, np.max(K), O,), dtype=object).tolist()
            for ov_idx, ovehicle in enumerate(ovehicles):
                for latent_idx in range(ovehicle.n_states):
                    for t in range(T):
                        yaws = ovehicle.pred_yaws[latent_idx][:,t]
                        vertices_k = vertices[t][latent_idx][ov_idx]
                        mean_theta_k = np.mean(yaws)
                        A_union_k, b_union_k = get_approx_union(mean_theta_k, vertices_k)
                        A_union[t][latent_idx][ov_idx] = A_union_k
                        b_union[t][latent_idx][ov_idx] = b_union_k
        
            return A_union, b_union
        elif self.__agent_type == "mcc" or self.__agent_type == "rmcc":
            N_traj, T, O = params.N_traj, self.__params.T, params.O
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
        """
        vertices = self.__compute_vertices(params, ovehicles)
        A_unions, b_unions = self.__compute_overapproximations(vertices, params, ovehicles)
        segs_polytopes, goal = self.compute_segs_polytopes_and_goal(params)
        
        """Optimize all trajectories if MCC, else partial trajectories."""
        # params.N_select params.N_traj params.subtraj_indices
        N_select = params.N_select if self.__agent_type == "rmcc" else params.N_traj

        """Common to MCC+RMCC: setup CPLEX variables"""
        L, T, K, O, Gamma, nu, nx = self.__params.L, self.__params.T, \
                params.K, params.O, self.__params.Gamma, self.__params.nu, self.__params.nx
        n_segs = len(segs_polytopes)
        u_max = self.__params.u_max
        model = docplex.mp.model.Model(name='obstacle_avoidance')
        # set up controls variables for each trajectory
        u = np.array(model.continuous_var_list(N_select*T*nu, lb=-u_max, ub=u_max, name='control'))
        Delta = np.array(model.binary_var_list(O*L*N_select*T, name='delta')).reshape(N_select, T, O, L)
        Omicron = np.array(model.binary_var_list(N_select*n_segs*T, name="omicron"),
                dtype=object).reshape(N_select, n_segs, T)

        """Common to MCC+RMCC: compute state from input variables"""
        # U has shape (N_select, T*nu)
        U = u.reshape(N_select, -1)
        # Gamma has shape (nx*(T + 1), nu*T) so X has shape (N_select, nx*(T + 1))
        X = util.obj_matmul(U, Gamma.T)
        # X, U have shapes (N_select, T, nx) and (N_select, T, nu) resp.
        X = (X + params.States_free_init).reshape(N_select, -1, nx)[..., 1:, :]
        U = U.reshape(N_select, -1, nu)

        """Common to MCC+RMCC: apply dynamics constraints to trajectories"""
        for _U, _X in zip(U, X):
            # _X, _U have shapes (T, nx) and (T, nu) resp.
            v_x, v_y = _X[..., 2], _X[..., 3]
            u_x, u_y = _U[..., 0], _U[..., 1]
            model.add_constraints(self.compute_velocity_constraints(v_x, v_y))
            model.add_constraints(self.compute_acceleration_constraints(u_x, u_y))

        """Common to MCC+RMCC: apply road boundaries constraints to trajectories"""
        M_big, T, diag = self.__params.M_big, self.__params.T, self.__params.diag
        for n in range(N_select):
            # for each trajectory
            for t in range(T):
                for seg_idx, (A, b) in enumerate(segs_polytopes):
                    lhs = util.obj_matmul(A, X[n,t,:2]) - np.array(M_big*(1 - Omicron[n,seg_idx,t]))
                    rhs = b + diag
                    model.add_constraints(
                            [l <= r for (l,r) in zip(lhs, rhs)])
                model.add_constraint(np.sum(Omicron[n,:,t]) >= 1)
        
        """Apply collision constraints"""
        traj_indices = params.subtraj_indices if self.__agent_type == "rmcc" else range(N_select)
        M_big, T, diag = self.__params.M_big, self.__params.T, self.__params.diag
        for n, i in enumerate(traj_indices):
            # select outerapprox. by index i
            # if MCC then iterates through every combination of obstacles
            A_union, b_union = A_unions[i], b_unions[i]
            for t in range(T):
                # for each timestep
                As, bs = A_union[t], b_union[t]
                for o, (A, b) in enumerate(zip(As, bs)):
                    lhs = util.obj_matmul(A, X[n,t,:2]) + M_big*(1 - Delta[n,t,o])
                    rhs = b + diag
                    model.add_constraints([l >= r for (l,r) in zip(lhs, rhs)])
                    model.add_constraint(np.sum(Delta[n,t,o]) >= 1)
        
        """Common to MCC+RMCC: set up coinciding constraints"""
        for t in range(0, self.__n_coincide):
            for x1, x2 in util.pairwise(X[:,t]):
                model.add_constraints([l == r for (l, r) in zip(x1, x2)])

        """Common to MCC+RMCC: set objective and run solver"""
        start = params.x0[:2]
        cost = np.sum((X[:,-1,:2] - goal)**2)
        model.minimize(cost)
        # model.print_information()
        if self.log_cplex:
            model.parameters.mip.display = 5
            s = model.solve(log_output=True)
        else:
            model.solve()
        # model.print_solution()

        f = lambda x: x if isinstance(x, numbers.Number) else x.solution_value
        U_star = util.obj_vectorize(f, U)
        cost = cost.solution_value
        X_star = util.obj_vectorize(f, X)
        return util.AttrDict(cost=cost, U_star=U_star, X_star=X_star,
                A_unions=A_unions, b_unions=b_unions, vertices=vertices,
                start=start, goal=goal)

    def do_single_control(self, params, ovehicles):
        """Decide parameters.

        TODO: acceleration limit is hardcoded to 8 m/s^2
        TODO finish description 
        """
        vertices = self.__compute_vertices(params, ovehicles)
        A_union, b_union = self.__compute_overapproximations(vertices, params, ovehicles)
        segs_polytopes, goal = self.compute_segs_polytopes_and_goal(params)

        """Apply motion planning problem"""
        n_segs = len(segs_polytopes)
        L, T, K, Gamma, nu, nx = self.__params.L, self.__params.T, params.K, \
                self.__params.Gamma, self.__params.nu, self.__params.nx
        u_max = self.__params.u_max
        model = docplex.mp.model.Model(name="proposed_problem")
        u = np.array(model.continuous_var_list(nu*T, lb=-u_max, ub=u_max, name='u'),
                dtype=object)
        Delta = np.array(model.binary_var_list(L*np.sum(K)*T, name='delta'),
                dtype=object).reshape(np.sum(K), T, L)
        Omicron = np.array(model.binary_var_list(n_segs*T, name="omicron"),
                dtype=object).reshape(n_segs, T)
        
        X = (params.States_free_init + util.obj_matmul(Gamma, u)).reshape(T + 1, nx)
        X = X[1:]
        U = u.reshape(T, nu)

        """Apply motion dynamics constraints"""
        model.add_constraints(self.compute_velocity_constraints(X[:, 2], X[:, 3]))
        model.add_constraints(self.compute_acceleration_constraints(U[:, 0], U[:, 1]))

        """Apply road boundaries constraints"""
        M_big, T, diag = self.__params.M_big, self.__params.T, self.__params.diag
        for t in range(T):
            for seg_idx, (A, b) in enumerate(segs_polytopes):
                lhs = util.obj_matmul(A, X[t, :2]) - np.array(M_big*(1 - Omicron[seg_idx, t]))
                rhs = b + diag
                model.add_constraints(
                        [l <= r for (l,r) in zip(lhs, rhs)])
            model.add_constraint(np.sum(Omicron[:, t]) >= 1)

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
        start = params.x0[:2]
        cost = (X[-1, 0] - goal[0])**2 + (X[-1, 1] - goal[1])**2
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
                start=start, goal=goal)

    def __plot_scenario(self, pred_result, ovehicles, params, ctrl_result):
        filename = f"agent{self.__ego_vehicle.id}_frame{params.frame}_lcss_control"
        if self.__agent_type == "oa":
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
            ego_bbox = [lon, lat]
            params.update(self.__params)
            plot_lcss_prediction(pred_result, ovehicles, params, ctrl_result,
                    self.__prediction_horizon, ego_bbox, filename=filename)
        elif self.__agent_type == "mcc":
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
            ego_bbox = [lon, lat]
            params.update(self.__params)
            plot_multiple_coinciding_controls(pred_result, ovehicles, params,
                    ctrl_result, ego_bbox, filename=filename)
        elif self.__agent_type == "rmcc":
            pass
        else:
            raise NotImplementedError()

    # @profile(sort_by='cumulative', lines_to_print=50, strip_dirs=True)
    def __compute_prediction_controls(self, frame):
        pred_result = self.do_prediction(frame)
        ovehicles = self.make_ovehicles(pred_result)
        params = self.make_local_params(frame, ovehicles)
        if self.__agent_type == "oa":
            ctrl_result = self.do_single_control(params, ovehicles)
        elif self.__agent_type == "mcc" or self.__agent_type == "rmcc":
            ctrl_result = self.do_multiple_control(params, ovehicles)
        else:
            raise NotImplementedError()

        """use control input next round for warm starting."""
        # self.U_warmstarting = ctrl_result.U_star

        """Get trajectory"""
        trajectory = []
        velocity = []
        # X has shape (T+1, nx)
        if self.__agent_type == "oa":
            X = np.concatenate((params.x0[None], ctrl_result.X_star))
            n_steps = X.shape[0]
        elif self.__agent_type == "mcc" or self.__agent_type == "rmcc":
            X = np.concatenate((params.x0[None], ctrl_result.X_star[0]))
            n_steps = self.__n_coincide + 1
            if self.log_agent:
                logging.info(f"Optimized {params.N_traj} "
                        f"trajectories avoiding {params.O} vehicles.")
        else:
            raise NotImplementedError()
        headings = []
        for t in range(1, n_steps):
            x, y = X[t, :2]
            y = -y # flip about x-axis again to move back to UE coordinates
            yaw = np.arctan2(X[t, 1] - X[t - 1, 1], X[t, 0] - X[t - 1, 0])
            headings.append(yaw)
            # flip about x-axis again to move back to UE coordinates
            yaw = np.rad2deg(util.reflect_radians_about_x_axis(yaw))
            transform = carla.Transform(carla.Location(x=x, y=y),carla.Rotation(yaw=yaw))
            trajectory.append(transform)
            velocity.append(math.sqrt(X[t, 2]**2 + X[t, 3]**2))

        if self.plot_scenario:
            """Plot scenario"""
            ctrl_result.headings = headings
            self.__plot_scenario(pred_result, ovehicles, params, ctrl_result)
        if self.plot_simulation:
            """Save planned trajectory for final plotting"""
            if self.__agent_type == "oa":
                self.__plot_simulation_data.planned_trajectories[frame] = np.concatenate(
                        (params.x0[None], ctrl_result.X_star))
                self.__plot_simulation_data.planned_controls[frame] = ctrl_result.U_star
            elif self.__agent_type == "mcc" or self.__agent_type == "rmcc":
                N_select = params.N_select if self.__agent_type == "rmcc" else params.N_traj
                self.__plot_simulation_data.planned_trajectories[frame] = np.concatenate(
                    (
                        np.repeat(params.x0[None], N_select, axis=0)[:, None],
                        ctrl_result.X_star
                    ), axis=1
                )
                self.__plot_simulation_data.planned_controls[frame] = ctrl_result.U_star
            else:
                pass
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
                trajectory, velocity = self.__compute_prediction_controls(frame)
                self.__local_planner.set_plan(trajectory,
                        self.__scene_config.record_interval, velocity=velocity)
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
