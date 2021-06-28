import sys
import math
from enum import Enum

import numpy as np
import pygame
from pygame.locals import *
import control
import control.matlab
import docplex.mp
import docplex.mp.model

import utility as util
import carla

# TODO: link this properly
from agents.tools.misc import is_within_distance_ahead, is_within_distance, compute_distance
from agents.navigation.controller import VehiclePIDController
from agents.navigation.local_planner_behavior import LocalPlanner
from agents.tools.misc import distance_vehicle, draw_waypoints

try:
    # trajectron-plus-plus/experiments/nuScenes
    from helper import load_model
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/experiments/nuScenes")

try:
    # trajectron-plus-plus/trajectron
    # from environment import Environment, Scene, Node
    # from environment import GeometricMap, derivative_of
    from model.dataset import get_timesteps_data
    # from model.components import *
    # from model.model_utils import *
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")


from generate import AbstractDataCollector
from generate import create_semantic_lidar_blueprint, get_all_vehicle_blueprints
from generate.scene import NaiveMapQuerier, OnlineConfig, SceneBuilder
from generate.scene.v2_1.trajectron_scene import TrajectronPlusPlusSceneBuilder


def generate_vehicle_latents(
            eval_stg, scene, timesteps,
            num_samples = 200, ph = 8,
            z_mode=False, gmm_mode = False, full_dist = False, all_z_sep = False):
    # Trajectron.predict() arguments
    min_future_timesteps = 0
    min_history_timesteps = 1

    node_type = eval_stg.env.NodeType.VEHICLE
    if node_type not in eval_stg.pred_state:
        raise Exception("fail")

    model = eval_stg.node_models_dict[node_type]

    # Get Input data for node type and given timesteps
    batch = get_timesteps_data(env=eval_stg.env, scene=scene, t=timesteps, node_type=node_type,
                               state=eval_stg.state,
                               pred_state=eval_stg.pred_state, edge_types=model.edge_types,
                               min_ht=min_history_timesteps, max_ht=eval_stg.max_ht,
                               min_ft=min_future_timesteps,
                               max_ft=min_future_timesteps, hyperparams=eval_stg.hyperparams)
    
    # There are no nodes of type present for timestep
    if batch is None:
        raise Exception("fail")

    (first_history_index,
     x_t, y_t, x_st_t, y_st_t,
     neighbors_data_st,
     neighbors_edge_value,
     robot_traj_st_t,
     map), nodes, timesteps_o = batch

    x = x_t.to(eval_stg.device)
    x_st_t = x_st_t.to(eval_stg.device)
    if robot_traj_st_t is not None:
        robot_traj_st_t = robot_traj_st_t.to(eval_stg.device)
    if type(map) == torch.Tensor:
        map = map.to(eval_stg.device)

    # MultimodalGenerativeCVAE.predict() arguments
    inputs = x
    inputs_st = x_st_t
    first_history_indices = first_history_index
    neighbors = neighbors_data_st
    neighbors_edge_value = neighbors_edge_value
    robot = robot_traj_st_t
    prediction_horizon = ph

    mode = ModeKeys.PREDICT

    x, x_nr_t, _, y_r, _, n_s_t0 = model.obtain_encoded_tensors(mode=mode,
                                                               inputs=inputs,
                                                               inputs_st=inputs_st,
                                                               labels=None,
                                                               labels_st=None,
                                                               first_history_indices=first_history_indices,
                                                               neighbors=neighbors,
                                                               neighbors_edge_value=neighbors_edge_value,
                                                               robot=robot,
                                                               map=map)

    model.latent.p_dist = model.p_z_x(mode, x)
    latent_probs = model.latent.get_p_dist_probs() \
            .cpu().detach().numpy()
    latent_probs = np.squeeze(latent_probs)

    z, num_samples, num_components = model.latent.sample_p(num_samples,
                                                          mode,
                                                          most_likely_z=z_mode,
                                                          full_dist=full_dist,
                                                          all_z_sep=all_z_sep)
    
    _, predictions = model.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode)

    z = z.cpu().detach().numpy()
    # z has shape (number of samples, number of vehicles, number of latent values)
    # z[i,j] gives the latent for sample i of vehicle j
#     print(z.shape)
    
    predictions = predictions.cpu().detach().numpy()
    # predictions has shape (number of samples, number of vehicles, prediction horizon, D)
#     print(predictions.shape)

    predictions_dict = dict()
    for i, ts in enumerate(timesteps_o):
        if ts not in predictions_dict.keys():
            predictions_dict[ts] = dict()
        predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))
    
    z = np.swapaxes(np.argmax(z, axis=-1), 0, 1)
    predictions = np.swapaxes(predictions, 0, 1)
        
    return z, predictions, nodes, predictions_dict, latent_probs


class OVehicle(object):

    @classmethod
    def from_trajectron(cls, node_id, ground_truth, past,
            latent_pmf, predictions, filter_pmf=0.1):
        """
        Parameters
        ==========
        ground_truth : np.array
            Array of ground truth trajectories of shape (T_gt_v, 2)
            T_gt_v is variable
        past : np.array
            Array of past trajectories of shape (T_past_v, 2)
            T_past_v is variable
        latent_pmf : np.array
            Array of past trajectories of shape (latent_states)
            Default settings in Trajectron++ sets latent_states to 25
        predictions : list of np.array
            List of predictions indexed by latent value.
            Each set of prediction corresponding to a latent is size (number of preds, T_v, 2)
        """
        n_states = len(predictions)
        n_predictions = sum([p.shape[0] for p in pred_positions])
        
        """Heejin's control code"""
        pos_last = past[-1]
        pos_prev = past[-4] # index this?
        p0 = pos_last
        yaw0 = np.arctan2(
                pos_last[1] - pos_prev[1],
                pos_last[0] - pos_prev[0])
        # Masking to get relevant predictions.
        latent_mask = np.argwhere(latent_pmf > filter_pmf).ravel()
        masked_n_predictions = 0
        masked_n_states = latent_mask.size
        masked_pred_positions = []
        masked_pred_yaws = []
        masked_init_center = np.zeros((masked_n_states, 2))

        for latent_idx, latent_val in enumerate(latent_mask):
            """Preprocess the predictions that correspond to the latents in the mask"""
            ps = predictions[latent_val]
            n_p = ps.shape[0]
            yaws = np.zeros((n_p, T))
            # diff = (ps[:,0,1] - pos_last[1])**2  - ( ps[:,0,0] - pos_last[0])**2
            # TODO: skipping the i_keep_yaw code
            yaws[:,0] = np.arctan2(ps[:,0,1] - pos_last[1], ps[:,0,0] - pos_last[0])
            for t in range(2, T):
                # diff = (ps[:,t,1] - ps[:,t-1,1])**2  - (ps[:,t,0] - ps[:,t-1,0])**2
                # TODO: skipping the i_keep_yaw code
                yaws[:,t] = np.arctan2(ps[:,t,1] - ps[:,t-1,1], ps[:,t,0] - ps[:,t-1,0])
            masked_pred_positions.append(ps)
            masked_pred_yaws.append(yaws)
            masked_n_predictions += n_p
            masked_init_center[latent_idx] = np.mean(ps[:, T-1], axis=0)
        latent_neg_mask = np.in1d(np.arange(n_states), latent_mask, invert=True)
        latent_neg_mask = np.arange(n_states)[latent_neg_mask]
        for latent_val in latent_neg_mask:
            """Group rare latent values to the closest common latent variable"""
            ps = predictions[latent_val]
            if ps.size == 0:
                continue
            n_p = ps.shape[0]
            yaws = np.zeros((n_p, T))
            # TODO: skipping the i_keep_yaw code
            yaws[:,0] = np.arctan2(ps[:,0,1] - pos_last[1], ps[:,0,0] - pos_last[0])
            for t in range(2, T):
                # TODO: skipping the i_keep_yaw code
                yaws[:,t] = np.arctan2(ps[:,t,1] - ps[:,t-1,1], ps[:,t,0] - ps[:,t-1,0])
            
            dist = scipy.spatial.distance_matrix(ps[:,T-1,:], masked_init_center)
            p_cluster_ids = np.argmin(dist, axis=1)
            for idx in range(masked_n_states):
                tmp_ps = ps[p_cluster_ids == idx]
                if tmp_ps.size == 0:
                    continue
                tmp_yaws = yaws[p_cluster_ids == idx]
                masked_pred_positions[idx] = np.concatenate(
                        (masked_pred_positions[idx], tmp_ps,))
                masked_pred_yaws[idx] = np.concatenate(
                        (masked_pred_yaws[idx], tmp_yaws,))
            masked_n_predictions += n_p

        masked_latent_pmf = np.zeros(latent_mask.shape)
        for idx in range(masked_n_states):
            """Recreate the PMF"""
            n_p = masked_pred_positions[idx].shape[0]
            masked_latent_pmf[idx] = n_p / float(masked_n_predictions)
        
        return cls(T, past, ground_truth, masked_latent_pmf, masked_pred_positions,
                masked_pred_yaws, masked_init_center)

    def __init__(self, T, past, ground_truth, latent_pmf, pred_positions,
            pred_yaws, init_center):
        self.T = T
        self.past = past
        self.ground_truth = ground_truth
        self.latent_pmf = latent_pmf
        self.pred_positions = pred_positions
        self.pred_yaws = pred_yaws
        self.init_center = init_center
        pos_last = past[-1]
        pos_prev = past[-4] # index this?
        self.p0 = pos_last
        self.yaw0 = np.arctan2(
                pos_last[1] - pos_prev[1],
                pos_last[0] - pos_prev[0])
        self.n_states = self.latent_pmf.size
        self.n_predictions = sum([p.shape[0] for p in self.pred_positions])


def get_vertices_from_center(center, heading, lw):
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

def get_ovehicle_color_set():
    OVEHICLE_COLORS = [
        clr.LinearSegmentedColormap.from_list('ro', ['red', 'orange'], N=256),
        clr.LinearSegmentedColormap.from_list('gy', ['green', 'yellow'], N=256),
        clr.LinearSegmentedColormap.from_list('bp', ['blue', 'purple'], N=256),
    ]
    ovehicle_colors = []
    for ov_colormap in OVEHICLE_COLORS:
        ov_colors = ov_colormap(np.linspace(0,1,5))
        ovehicle_colors.append(ov_colors)
    return ovehicle_colors

class LCSSHighLevelAgent(AbstractDataCollector):
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
            scene_config=OnlineConfig()):
        
        self.__ego_vehicle = ego_vehicle
        self.__map_reader = map_reader

        # __first_frame : int
        #     First frame in simulation. Used to find current timestep.
        self.__first_frame = None

        self.__scene_builder = None
        self.__scene_config = scene_config
        self.__predict_interval = 6
        self.__n_burn_interval = 4
        self.__prediction_horizon = 8
        self.__n_predictions = 100
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

        self.__local_planner = LocalPlanner(self)

    def start_sensor(self):
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.__sensor.listen(lambda image: DataCollector.parse_image(weak_self, image))
    
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

    @property
    def vehicle(self):
        return self.__ego_vehicle
    
    @property
    def T(self):
        return self.__prediction_horizon

    def do_first_step(self, first_frame):
        self.__first_frame = first_frame
        self.__scene_builder = TrajectronPlusPlusSceneBuilder(
            self,
            self.__map_reader,
            self.__ego_vehicle,
            self.__other_vehicles,
            self.__lidar_feeds,
            "test",
            self.__first_frame,
            scene_config=self.__scene_config,
            debug=True)

    def do_prediction(self, frame):

        """Construct online scene"""
        scene = self.__scene_builder.get_scene()

        """Extract Predictions"""
        frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
        timestep = frame_id # we use this as the timestep
        timesteps = np.array([timestep])
        with torch.no_grad():
            z, predictions, nodes, predictions_dict, latent_probs = generate_vehicle_latents(
                    self.__eval_stg, scene, timesteps, num_samples=200, ph=8,
                    z_mode=False, gmm_mode=False, full_dist=False, all_z_sep=False)

        _, past_dict, ground_truth_dict = \
            prediction_output_to_trajectories(
                predictions, dt=scene.dt, max_h=10, ph=self.__prediction_horizon, map=None)
        return util.AttrDict(scene=scene, timestep=timestep, nodes=nodes,
                predictions=predictions, z=z,
                past_dict=past_dict, ground_truth_dict=ground_truth_dict)
        
    def make_ovehicles(self, result):
        scene, timestep, nodes = result.scene, result.timestep, result.nodes
        predictions, latent_probs, z = result.predictions, result.latent_probs, result.z
        
        """Preprocess predictions"""
        minpos = np.array([scene.x_min, scene.y_min])
        ovehicles = []
        for idx, node in enumerate(nodes):
            if node.id == 'ego':
                continue
            veh_gt         = ground_truth_dict[timestep][node] + minpos
            veh_past       = past_dict[timestep][node] + minpos
            veh_predict    = predictions[idx] + minpos
            veh_latent_pmf = latent_probs[idx]
            zn = z[idx]
            veh_latent_predictions = [[] for x in range(25)]
            for jdx, p in enumerate(veh_predict):
                veh_latent_predictions[zn[jdx]].append(p)
            ovehicle = OVehicle(repr(node), veh_gt, veh_past,
                    veh_latent_pmf, veh_latent_predictions)
            ovehicles.append(ovehicle)
        
        return ovehicles
    
    def make_highlevel_params(self, ovehicles):
        p_0_x, p_0_y, _ = carlautil.actor_to_location_ndarray(self.__ego_vehicle)
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(self.__ego_vehicle)
        
        """Get LCSS parameters"""
        # TODO: refactor this long chain
        params = util.AttrDict()
        params.Ts = 0.5
        # A, sys.A both have shape (4, 4)
        A = np.diag([1, 1], k=2)
        # B, sys.B both have shape (4, 2)
        B = np.concatenate((np.diag([0,0]), np.diag([1,1]),))
        # C has shape (2, 4)
        C = np.concatenate((np.diag([1,1]), np.diag([0,0]),), axis=1)
        # D has shape (2, 2)
        D = np.diag([0, 0])
        sys = control.matlab.c2d(control.matlab.ss(A, B, C, D), params.Ts)
        params.A = np.array(sys.A)
        params.B = np.array(sys.B)
        # number of state variables x, number of input variables u
        # nx = 4, nu = 2
        params.nx, params.nu = sys.B.shape
        # TODO: remove car, truck magic numbers
        # only truck.d is used
        params.car = util.AttrDict()
        params.car.d = np.array([4.5, 2.5])
        # truck should be renamed to car, and only truck.d is used
        params.truck = util.AttrDict()
        params.truck.d = [4.5, 2.5]
        # params.x0 : np.array
        #   Initial state
        params.x0 = np.array([p_0_x, p_0_y, v_0_x, v_0_y])
        params.diag = np.sqrt(params.car.d[1]**2 + params.car.d[0]**2) / 2.
        # TODO: remove constraint of v, u magic numbers
        # Vehicle dynamics restrictions
        params.vmin = np.array([0/3.6, -20/3.6]) # lower bounds
        params.vmax = np.array([80/3.6, 20/3.6]) # upper bounds
        params.umin = np.array([-10, -5])
        params.umax = np.array([3, 5])
        # Prediction parameters
        params.T = self.__prediction_horizon
        params.O = len(ovehicles) # number of obstacles
        params.L = 4 # number of faces of obstacle sets
        params.K = np.zeros(params.O, dtype=int)
        for idx, vehicle in enumerate(ovehicles):
            params.K[idx] = vehicle.n_states

        """State and input constraints (Box constraints)"""
        params.umin_bold = np.tile(params.umin, params.T)
        params.umax_bold = np.tile(params.umax, params.T)
    
        """
        3rd and 4th states have COUPLED CONSTRAINTS
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %       x4 - c1*x3 <= - c3
        %       x4 - c1*x3 >= - c2
        %       x4 + c1*x3 <= c2
        %       x4 + c1*x3 >= c3
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        vmin, vmax = params.vmin, params.vmax
        # they work only assuming vmax(2) = -vmin(2)
        params.c1 = vmax[1] / (0.5*(vmax[0] - vmin[0]))
        params.c2 = params.c1 * vmax[0]
        params.c3 = params.c1 * vmin[0]
    
        """
        Inputs have COUPLED CONSTRAINTS
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %       u2 - t1*u1 <= - t3
        %       u2 - t1*u1 >= - t2
        %       u2 + t1*u1 <= t2
        %       u2 + t1*u1 >= t3
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        umin, umax = params.umin, params.umax
        params.t1 = umax[1]/(0.5*(umax[0] - umin[0]))
        params.t2 = params.t1 * umax[0]
        params.t3 = params.t1 * umin[0]

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
        Xx = np.linalg.solve(Abar, Bbar)
        # Gamma has shape (nx*(T + 1), nu*T) as Abar\Bbar has shape (nx*T, nu*T)
        Gamma = np.concatenate((np.zeros((nx, T*nu,)), np.linalg.solve(Abar, Bbar),))
        params.Abar = Abar
        params.Bbar = Bbar
        params.Gamma = Gamma
        A, T, x0 = params.A, params.T, params.x0
        # States_free_init has shape (nx*(T+1))
        params.States_free_init = np.concatenate([(A**t) @ x0 for t in range(T+1)])
        return params

    def do_highlevel_control(self, params, ovehicles):
        """Decide parameters"""
        # TODO: assign eps^k_o and beta^k_o to the vehicles.
        # Skipping that for now.
    
        """Compute the approximation of the union of obstacle sets"""
        # Find vertices of sampled obstacle sets
        T, K, n_ov, truck = params.T, params.K, params.O, params.truck
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
                                ps[k,t], yaws[k,t], truck.d)
        
        plot_vertices = False
        if plot_vertices:
            t = 7
            # for ovehicle in scene.ovehicles:
            X = vertices[t][0][1][:,0:2].T
            plt.scatter(X[0], X[1], c='r', s=2)
            X = vertices[t][0][1][:,2:4].T
            plt.scatter(X[0], X[1], c='b', s=2)
            X = vertices[t][0][1][:,4:6].T
            plt.scatter(X[0], X[1], c='g', s=2)
            X = vertices[t][0][1][:,6:8].T
            plt.scatter(X[0], X[1], c='orange', s=2)
            plt.gca().set_aspect('equal')
            plt.show()

        # TODO: time this
        # t_overapprox_start = tic

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
            # for ovehicle in scene.ovehicles:
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
        big_M = 200 # conservative upper-bound

        x1 = x_future[nx::nx]
        x2 = x_future[nx + 1::nx]
        x3 = x_future[nx + 2::nx]
        x4 = x_future[nx + 3::nx]

        # 3rd and 4th states have COUPLED CONSTRAINTS
        #       x4 - c1*x3 <= - c3
        #       x4 - c1*x3 >= - c2
        #       x4 + c1*x3 <= c2
        #       x4 + c1*x3 >= c3
        c1, c2, c3 = params.c1, params.c2, params.c3
        model.add_constraints([z <= -c3 for z in  x4 - c1*x3])
        model.add_constraints([z <=  c2 for z in -x4 + c1*x3])
        model.add_constraints([z <=  c2 for z in  x4 + c1*x3])
        model.add_constraints([z <= -c3 for z in -x4 - c1*x3])

        u1 = u[0::nu]
        u2 = u[1::nu]

        # Inputs have COUPLED CONSTRAINTS:
        #       u2 - t1*u1 <= - t3
        #       u2 - t1*u1 >= - t2
        #       u2 + t1*u1 <= t2
        #       u2 + t1*u1 >= t3
        t1, t2, t3 = params.t1, params.t2, params.t3
        model.add_constraints([z <= -t3 for z in  u2 - t1*u1])
        model.add_constraints([z <=  t2 for z in -u2 + t1*u1])
        model.add_constraints([z <=  t2 for z in  u2 + t1*u1])
        model.add_constraints([z <= -t3 for z in -u2 - t1*u1])

        X = np.stack([x1, x2])
        Rot = evehicle.Rot
        X = obj_matmul(Rot, X).T + evehicle.p0

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
        
        # cost = x2[-1]**2 + x3[-1]**2 + x4[-1]**2 - 0.01*x1[-1]
        # start from spawn point 2
        p_x, p_y, _ = carlautil.actor_to_location_ndarray(
                self.__ego_vehicle)
        final_x, final_y = p_x + 50, p_y
        cost = (x3[-1] - final_x)**2 + (x4[-1] - final_y)**2
        model.minimize(cost)
        # model.print_information()
        s = model.solve()
        # model.print_solution()

        u_star = np.array([ui.solution_value for ui in u])
        cost = cost.solution_value
        x1 = np.array([x1i.solution_value for x1i in x1])
        x2 = np.array([x2i.solution_value for x2i in x2])
        X = np.stack((x1, x2))
        X_star = (Rot @ X).T + evehicle.p0
        X_star = np.concatenate((evehicle.p0[None], X_star,))
        return util.AttrDict(cost=cost, u_star=u_star, X_star=X_star,
                A_union=A_union, b_union=b_union, vertices=vertices)

    def __capture_prediction_step(frame):
        pred_result = self.do_prediction(frame)
        ovehicles = self.make_ovehicles(pred_result)
        params = self.make_highlevel_params(ovehicles):
        ctrl_result = self.do_highlevel_control(params, ovehicles)

        _, heading, _ = carlautil.actor_to_rotation_ndarray(self.__ego_vehicle)

        """Plots for paper"""
        ovehicle_colors = get_ovehicle_color_set()
        fig, ax = plt.subplots()
        t = self.__prediction_horizon - 1

        # Plot ego vehicle
        ax.plot(result.X_star[:, 0], result.X_star[:, 1], 'k-o')

        # Get vertices of EV and plot its bounding box
        vertices = get_vertices_from_center(
                result.X_star[-1], heading, params.truck.d)
        bb = patches.Polygon(vertices.reshape((-1,2,)),
                closed=True, color='k', fc='none')
        ax.add_patch(bb)

        # Plot other vehicles
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                color = ovehicle_colors[ov_idx][latent_idx]
                # ps = ovehicle.pred_positions[latent_idx]

                # Plot overapproximation
                A = result.A_union[t][latent_idx][ov_idx]
                b = result.b_union[t][latent_idx][ov_idx]
                plot_h_polyhedron(ax, A, b, ec=color, alpha=1)

                # Plot vertices
                vertices = result.vertices[t][latent_idx][ov_idx]
                X = vertices[:,0:2].T
                ax.scatter(X[0], X[1], color=color, s=2)
                X = vertices[:,2:4].T
                ax.scatter(X[0], X[1], color=color, s=2)
                X = vertices[:,4:6].T
                ax.scatter(X[0], X[1], color=color, s=2)
                X = vertices[:,6:8].T
                ax.scatter(X[0], X[1], color=color, s=2)

        ax.set_aspect('equal')
        ax.set_facecolor("grey")
        plt.show()
        raise Exception("ASDF")

    def capture_step(self, frame):
        if self.__first_frame is None:
            self.do_first_step(frame)
        
        self.__scene_builder.capture_trajectory(frame)
        
        """Initially collect data without doing anything to the vehicle."""
        if frame - self.__first_frame < self.__n_burn_frames:
            return

        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            frame_id = int((frame - self.__first_frame) / self.__scene_config.record_interval)
            if frame_id < self.__n_burn_interval:
                return
            if (frame_id - self.__n_burn_interval) % self.__predict_interval == 0:
                self.__capture_prediction_step(frame)

    def remove_scene_builder(self, first_frame):
        raise Exception(f"Can't remove scene builder from {util.classname(x)}.")

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
        for scene_builder in list(self.__scene_builders.values()):
            scene_builder.capture_lidar(image)

    # def get_speed(self):
    #     self.vehicle.get_velocity()
    #     return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    
    # def update_information(self):
    #     """Update agent's sensor data."""
    #     self.speed = get_speed(self.vehicle)
    #     self.speed_limit = self.vehicle.get_speed_limit()

class OnlineManager(object):
    
    def __init__(self):
        
        """Load model."""
        model_dir = 'models/20210622'
        model_name = 'models_19_Mar_2021_22_14_19_int_ee_me_ph8'
        model_path = os.path.join(model_dir, model_name)
        self.eval_stg, self.stg_hyp = load_model(
                model_path, eval_env, ts=20)#, device='cuda:0')

        """Get CARLA connectors"""
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.carla_map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.intersection_reader = IntersectionReader(
                self.world, self.carla_map, debug=self.args.debug)
        self.map_reader = NaiveMapQuerier(
                self.world, self.carla_map, debug=self.args.debug)
        self.online_config = OnlineConfig()

    
    def create_agents(self, params):
        spawn_points = self.carla_map.get_spawn_points()
        blueprint = self.world.get_blueprint_library().find('vehicle.audi.a2')
        spawn_point = spawn_points[params.ego_spawn_idx]
        params.ego_vehicle = self.world.spawn_actor(blueprint, spawn_point)

        other_vehicle_ids = []
        blueprints = get_all_vehicle_blueprints(world)
        for idx in other_spawn_ids:
            blueprint = np.random.choice(blueprints)
            spawn_point = spawn_points[idx]
            other_vehicle = self.world.spawn_actor(blueprint, spawn_point)
            other_vehicle.set_autopilot(True, self.traffic_manager.get_port())
            params.other_vehicles.append(other_vehicle)
            other_vehicle_ids.append(other_vehicle.id)

        """Get our high level APIs"""
        params.agent = LCSSHighLevelAgent(
                params.ego_vehicle,
                self.map_reader,
                other_vehicle_ids,
                self.eval_stg,
                scene_config=self.online_config)
        params.agent.start_sensor()
        return params

    def run(self):
        params = util.AttrDict(ego_spawn_idx=2, other_spawn_ids=[104, 102, 235],
                ego_vehicle=None, other_vehicles=[], agent=None)
        n_burn_frames = 3*self.online_config.record_interval
        try:
            params = self.create_agents(params)
            agent = params.agent

            for idx in range(n_burn_frames \
                    + 30*self.online_config.record_interval):
                frame = self.world.tick()
                agent = capture_step(frame)
        finally:
            if params.agent:
                params.agent.destroy()
            if params.ego_vehicle:
                params.ego_vehicle.destroy()
            for other_vehicle in params.other_vehicles:
                other_vehicle.destroy()


if __name__ == '__main__':
    mgr = OnlineManager()
    mgr.run()

# control = carla.Vehicle(
#         throttle=0.0, steer=0.0, brake=0.0, 
#         hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
# vehicle.apply_control(control)

# pygame.init()
# vec = pygame.math.Vector2  # 2 for two dimensional
 
# HEIGHT = 450
# WIDTH = 400
# ACC = 0.5
# FRIC = -0.12
# FPS = 60
 
# FramePerSec = pygame.time.Clock()
 
# displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("Game")
