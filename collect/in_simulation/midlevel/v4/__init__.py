"""
v4 is a modification of v3 that does random multiple coinciding control.
It randomly selects a subset of trajectories to optimize.
"""

# Built-in libraries
import logging
import collections
import weakref
import copy
import numbers

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
        plot_multiple_coinciding_controls, get_vertices_from_centers)
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
            n_coincide=6,
            scene_builder_cls=TrajectronPlusPlusSceneBuilder,
            scene_config=OnlineConfig(),
            log_cplex=False,
            log_agent=False,
            plot_scenario=False,
            plot_boundary=False,
            plot_vertices=False,
            plot_overapprox=False,
            **kwargs):
        assert control_horizon <= prediction_horizon
        assert n_coincide <= control_horizon
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
        self.__n_coincide = n_coincide
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
        self.log_cplex = log_cplex
        self.log_agent = log_agent
        self.plot_scenario = plot_scenario
        self.plot_boundary = plot_boundary
        self.plot_vertices = plot_vertices
        self.plot_overapprox = plot_overapprox

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
        # N_traj - number of planned trajectories possible to compute
        params.N_traj = np.prod(params.K)
        max_K = np.max(params.K)
        # TODO: figure out a way to specify number of subtrajectories
        params.N_select = min(int(1.5*max_K), params.N_traj)
        # subtraj_indices - the indices of the sub-trajectory to optimize for
        params.subtraj_indices = np.random.choice(
                np.arange(params.N_traj), size=params.N_select, replace=False)
        p_0_x, p_0_y, _ = carlautil.to_location_ndarray(
                self.__ego_vehicle, flip_y=True)
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(
                self.__ego_vehicle, flip_y=True)
        # x0 : np.array
        #   Initial state
        x0 = np.array([p_0_x, p_0_y, v_0_x, v_0_y])
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
            Has p_x = <p_x_1, p_x_2, ..., p_x_T> where timesteps T is the
            extent of motion plan to enforce constraints in the x axis.  
        p_y : np.array of docplex.mp.vartype.VarType
            Has p_y = <p_y_1, p_y_2, ..., p_y_T> where timesteps T is the
            extent of motion plan to enforce constraints in the y axis.
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
            Has v_x = <v_x_1, v_x_2, ..., v_x_T> where timesteps T is the
            extent of motion plan to enforce constraints in the x axis.  
        v_y : np.array of docplex.mp.vartype.VarType
            Has v_y = <v_y_1, v_y_2, ..., v_y_T> where timesteps T is the
            extent of motion plan to enforce constraints in the y axis.
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
        _, theta, _ = carlautil.actor_to_rotation_ndarray(
                self.__ego_vehicle, flip_y=True)
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

    def __compute_vertices(self, params, ovehicles):
        """Compute verticles from predictions."""
        T, K, O = self.__params.T, params.K, params.O
        vertices = np.empty((T, np.max(K), O,), dtype=object).tolist()
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
            Collection of A matrices of shape (N_traj, T, O, L, 2).
            Axis 2 (zero-based) is sorted by ovehicle.
        ndarray
            Collection of b vectors of shape (N_traj, T, O, L).
            Axis 2 (zero-based) is sorted by ovehicle.
        """
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

    def do_highlevel_control(self, params, ovehicles):
        """Decide parameters.

        Since we're doing multiple coinciding, we want to compute...
        TODO finish description 
        """
        vertices = self.__compute_vertices(params, ovehicles)
        A_unions, b_unions = self.__compute_overapproximations(vertices, params, ovehicles)
        
        """Apply motion planning problem"""
        # params.N_select params.N_traj params.subtraj_indices
        N_select, L, T, K, O, Gamma, nu, nx = params.N_select, self.__params.L, self.__params.T, \
                params.K, params.O, self.__params.Gamma, self.__params.nu, self.__params.nx
        model = docplex.mp.model.Model(name='obstacle_avoidance')
        # set up controls variables for each trajectory
        u = np.array(model.continuous_var_list(N_select*T*nu, lb=-8., ub=8., name='control'))
        Delta = np.array(model.binary_var_list(O*L*N_select*T, name='delta')).reshape(N_select, T, O, L)

        # U has shape (N_select, T*nu)
        U = u.reshape(N_select, -1)
        # Gamma has shape (nx*(T + 1), nu*T) so X has shape (N_select, nx*(T + 1))
        X = util.obj_matmul(U, Gamma.T)
        # X, U have shapes (N_select, T, nx) and (N_select, T, nu) resp.
        X = (X + params.States_free_init).reshape(N_select, -1, nx)[..., 1:, :]
        U = U.reshape(N_select, -1, nu)

        for _U, _X in zip(U, X):
            # _X, _U have shapes (T, nx) and (T, nu) resp.
            p_x, p_y = _X[..., 0], _X[..., 1]
            v_x, v_y = _X[..., 2], _X[..., 3]
            u_x, u_y = _U[..., 0], _U[..., 1]
            model.add_constraints(self.compute_boundary_constraints(p_x, p_y))
            model.add_constraints(self.compute_velocity_constraints(v_x, v_y))
            model.add_constraints(self.compute_acceleration_constraints(u_x, u_y))

        # set up obstacle constraints
        subtraj_indices, M_big, T, diag = params.subtraj_indices, self.__params.M_big, \
                self.__params.T, self.__params.diag
        for n, i in enumerate(subtraj_indices):
            # select outerapprox. by index i
            A_union, b_union = A_unions[i], b_unions[i]
            for t in range(T):
                # for each timestep
                As, bs = A_union[t], b_union[t]
                for o, (A, b) in enumerate(zip(As, bs)):
                    lhs = util.obj_matmul(A, X[n,t,:2]) + M_big*(1 - Delta[n,t,o])
                    rhs = b + diag
                    model.add_constraints([l >= r for (l,r) in zip(lhs, rhs)])
                    model.add_constraint(np.sum(Delta[n,t,o]) >= 1)

        # set up coinciding constraints
        for t in range(0, self.__n_coincide):
            for x1, x2 in util.pairwise(X[:,t]):
                model.add_constraints([l == r for (l,r) in zip(x1, x2)])

        # start from current vehicle position and minimize the objective
        p_x, p_y, _ = carlautil.to_location_ndarray(
                self.__ego_vehicle, flip_y=True)
        if self.__goal.is_relative:
            goal_x, goal_y = p_x + self.__goal.x, p_y + self.__goal.y
        else:
            goal_x, goal_y = self.__goal.x, self.__goal.y
        start = np.array([p_x, p_y])
        goal = np.array([goal_x, goal_y])
        cost = np.sum((X[:, -1, :2] - goal)**2)
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

    # @profile(sort_by='cumulative', lines_to_print=50, strip_dirs=True)
    def __compute_prediction_controls(self, frame):
        pred_result = self.do_prediction(frame)
        ovehicles = self.make_ovehicles(pred_result)
        params = self.make_local_params(ovehicles)
        ctrl_result = self.do_highlevel_control(params, ovehicles)

        """Get trajectory"""
        # X has shape (T+1, nx)
        X = np.concatenate((ctrl_result.start[None], ctrl_result.X_star[0, :, :2]))
        trajectory = []
        for t in range(1, self.__n_coincide + 1):
            x, y = X[t]
            y = -y # flip about x-axis again to move back to UE coordinates
            yaw = np.arctan2(X[t, 1] - X[t - 1, 1], X[t, 0] - X[t - 1, 0])
             # flip about x-axis again to move back to UE coordinates
            yaw = util.reflect_radians_about_x_axis(yaw)
            transform = carla.Transform(
                    carla.Location(x=x, y=y),
                    carla.Rotation(yaw=yaw))
            trajectory.append(transform)

        if self.log_agent:
            logging.info(f"Optimized {params.N_select}/{params.N_traj} trajectories avoiding {params.O} vehicles.")
            logging.info(f"Selected trajectory indices are: {params.subtraj_indices}")

        if self.plot_scenario:
            """Plot scenario"""
            filename = f"agent{self.__ego_vehicle.id}_frame{frame}_lcss_control"
            lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
            ego_bbox = [lon, lat]
            params.update(self.__params)
            plot_multiple_coinciding_controls(pred_result, ovehicles, params,
                    ctrl_result, ego_bbox, filename=filename)

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
            if frame_id < self.__n_burn_interval:
                """Initially collect data without doing any control to the vehicle."""
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
