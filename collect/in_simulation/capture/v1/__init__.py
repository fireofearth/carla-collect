# Built-in libraries
import os
import logging
import collections
from typing import overload
import weakref
import copy
import numbers
import math

# PyPI libraries
import numpy as np
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm
import matplotlib.patches as patches
import torch
import docplex.mp
import docplex.mp.model
import docplex.mp.utils

try:
    from utils.trajectory_utils import prediction_output_to_trajectories
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

from ...midlevel.util import compute_L4_outerapproximation
from ...midlevel.ovehicle import OVehicle
from ...midlevel.prediction import generate_vehicle_latents
from ...midlevel.plotting import PlotCluster
from ....generate import AbstractDataCollector
from ....generate import create_semantic_lidar_blueprint
from ....generate.map import MapQuerier
from ....generate.scene import OnlineConfig
from ....generate.scene.v3_2.trajectron_scene import TrajectronPlusPlusSceneBuilder

# Local libraries
import carla
import utility as util
import utility.npu
import carlautil
import carlautil.debug

def get_speed(vehicle):
    """Get velocity of vehicle in m/s."""
    v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(vehicle, flip_y=True)
    return np.sqrt(v_0_x ** 2 + v_0_y ** 2)

class CapturingAgent(AbstractDataCollector):
    
    Z_SENSOR_REL = 2.5

    def __create_segmentation_lidar_sensor(self):
        return self.__world.spawn_actor(
            create_semantic_lidar_blueprint(self.__world),
            carla.Transform(carla.Location(z=self.Z_SENSOR_REL)),
            attach_to=self.__ego_vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )

    def __init__(
        self,
        ego_vehicle,
        map_reader: MapQuerier,
        other_vehicle_ids,
        eval_stg,
        scene_builder_cls=TrajectronPlusPlusSceneBuilder,
        scene_config=OnlineConfig(),
        ##########
        # Sampling
        n_burn_interval=4,
        n_predictions=100,
        prediction_horizon=8,
        step_horizon=1,
        ##########
        **kwargs
    ):
        # __ego_vehicle : carla.Vehicle
        #   The vehicle to control in the simulator.
        self.__ego_vehicle = ego_vehicle
        # __map_reader : MapQuerier
        #   To query map data.
        self.__map_reader = map_reader
        # __eval_stg : Trajectron
        #   Prediction Model to generate multi-agent forecasts.
        self.__eval_stg = eval_stg
        # __n_predictions : int
        #   Number of predictions to generate on each control step.
        self.__n_predictions = n_predictions
        # __n_burn_interval : int
        #   Interval in prediction timesteps to skip prediction and control.
        self.__n_burn_interval = n_burn_interval
        # __prediction_horizon : int
        #   Number of predictions timesteps to predict other vehicles over.
        self.__prediction_horizon = prediction_horizon
        # __step_horizon : int
        #   Number of predictions steps to execute at each iteration of MPC.
        self.__step_horizon = step_horizon
        self.__scene_builder_cls = scene_builder_cls
        self.__scene_config = scene_config
        # __first_frame : int
        #   First frame in simulation. Used to find current timestep.
        self.__first_frame = None
        self.__world = self.__ego_vehicle.get_world()
        vehicles = self.__world.get_actors(other_vehicle_ids)
        # __other_vehicles : list of carla.Vehicle
        #     List of IDs of vehicles not including __ego_vehicle.
        #     Use this to track other vehicles in the scene at each timestep.
        self.__other_vehicles = dict(zip(other_vehicle_ids, vehicles))
        # __steptime : float
        #   Time in seconds taken to complete one step of MPC.
        self.__steptime = (
            self.__scene_config.record_interval
            * self.__world.get_settings().fixed_delta_seconds
        )
        # __sensor : carla.Sensor
        #     Segmentation sensor. Data points will be used to construct overhead.
        self.__sensor = self.__create_segmentation_lidar_sensor()
        # __lidar_feeds : collections.OrderedDict
        #     Where int key is frame index and value
        #     is a carla.LidarMeasurement or carla.SemanticLidarMeasurement
        self.__lidar_feeds = collections.OrderedDict()

        self.__plot_simulation_data = util.AttrDict(
            states=collections.OrderedDict(),
            bboxes=collections.OrderedDict(),
            vertices=collections.OrderedDict(),
            OK_Ab_union=collections.OrderedDict(),
        )

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
        if len(self.__plot_simulation_data.states) == 0:
            return
        filename = f"agent{self.__ego_vehicle.id}_clusters"
        plotter = PlotCluster(
            self.__map_reader.map_data,
            self.__plot_simulation_data.states,
            self.__plot_simulation_data.bboxes,
            self.__plot_simulation_data.vertices,
            self.__plot_simulation_data.OK_Ab_union,
            self.__prediction_horizon,
            filename=filename
        )
        # plotter.plot()
        # plotter.plot_overapprox_per_vehicle()
        plotter.plot_convexhull_per_vehicle()

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        self.__sensor.destroy()
        self.__sensor = None
        self.__plot_simulation()

    def do_prediction(self, frame):
        """Get processed scene object from scene builder,
        input the scene to a model to generate the predictions,
        and then return the predictions and the latents variables."""

        """Construct online scene"""
        scene = self.__scene_builder.get_scene()

        """Extract Predictions"""
        frame_id = int(
            (frame - self.__first_frame) / self.__scene_config.record_interval
        )
        timestep = frame_id  # we use this as the timestep
        timesteps = np.array([timestep])
        with torch.no_grad():
            (
                z,
                predictions,
                nodes,
                predictions_dict,
                latent_probs,
            ) = generate_vehicle_latents(
                self.__eval_stg,
                scene,
                timesteps,
                num_samples=self.__n_predictions,
                ph=self.__prediction_horizon,
                z_mode=False,
                gmm_mode=False,
                full_dist=False,
                all_z_sep=False,
            )

        _, past_dict, ground_truth_dict = prediction_output_to_trajectories(
            predictions_dict,
            dt=scene.dt,
            max_h=10,
            ph=self.__prediction_horizon,
            map=None,
        )
        return util.AttrDict(
            scene=scene,
            timestep=timestep,
            nodes=nodes,
            predictions=predictions,
            z=z,
            latent_probs=latent_probs,
            past_dict=past_dict,
            ground_truth_dict=ground_truth_dict,
        )

    def make_ovehicles(self, result):
        scene, timestep, nodes = result.scene, result.timestep, result.nodes
        predictions, latent_probs, z = result.predictions, result.latent_probs, result.z
        past_dict, ground_truth_dict = result.past_dict, result.ground_truth_dict

        """Preprocess predictions"""
        minpos = np.array([scene.x_min, scene.y_min])
        ovehicles = []
        for idx, node in enumerate(nodes):
            if node.id == "ego":
                lon, lat, _ = carlautil.actor_to_bbox_ndarray(
                    self.__ego_vehicle
                )
            else:
                lon, lat, _ = carlautil.actor_to_bbox_ndarray(
                    self.__other_vehicles[int(node.id)]
                )
            veh_bbox = np.array([lon, lat])
            veh_gt = ground_truth_dict[timestep][node] + minpos
            veh_past = past_dict[timestep][node] + minpos
            veh_predict = predictions[idx] + minpos
            veh_latent_pmf = latent_probs[idx]
            n_states = veh_latent_pmf.size
            zn = z[idx]
            veh_latent_predictions = [[] for x in range(n_states)]
            for jdx, p in enumerate(veh_predict):
                veh_latent_predictions[zn[jdx]].append(p)
            for jdx in range(n_states):
                veh_latent_predictions[jdx] = np.array(veh_latent_predictions[jdx])
            ovehicle = OVehicle.from_trajectron(
                node,
                self.__prediction_horizon,
                veh_gt,
                veh_past,
                veh_latent_pmf,
                veh_latent_predictions,
                bbox=veh_bbox,
            )
            # if node.id != "ego":
            #     vehicle = self.__other_vehicles[int(node.id)]
            #     pos = carlautil.to_location_ndarray(vehicle, flip_y=True)
            #     out = f"id: {node.id}, pos: {pos}, last: {ovehicle.past[-1]}, gt: {ovehicle.ground_truth}"
            #     logging.info(out)
                
            ovehicles.append(ovehicle)
        return ovehicles

    def get_current_velocity(self):
        """Get current velocity of vehicle in m/s."""
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(
            self.__ego_vehicle, flip_y=True
        )
        return np.sqrt(v_0_x ** 2 + v_0_y ** 2)

    def make_local_params(self, frame, ovehicles):
        """Get the local optimization parameters used for current MPC step."""

        """Get parameters to construct control and state variables."""
        params = util.AttrDict()
        params.frame = frame
        p_0_x, p_0_y, _ = carlautil.to_location_ndarray(self.__ego_vehicle, flip_y=True)
        _, psi_0, _ = carlautil.actor_to_rotation_ndarray(
            self.__ego_vehicle, flip_y=True
        )
        v_0_mag = self.get_current_velocity()
        x_init = np.array([p_0_x, p_0_y, psi_0, v_0_mag])
        initial_state = util.AttrDict(world=x_init, local=np.array([0, 0, 0, v_0_mag]))
        params.initial_state = initial_state

        """Get controls for other vehicles."""
        # O - number of total vehicles EV and OVs
        params.O = len(ovehicles)
        # K - for each o=1,...,O K[o] is the number of outer approximations for vehicle o
        params.K = np.zeros(params.O, dtype=int)
        for idx, vehicle in enumerate(ovehicles):
            params.K[idx] = vehicle.n_states
        return params


    def compute_vertices(self, params, ovehicles):
        """Compute verticles from predictions.
        
        Parameters
        ==========
        params : util.AttrDict
            Parameters for current MPC step.
        ovehicles : OVehicle
            Vehicle data to produce vertices from.
        
        Returns
        =======
        list
            Vertex sets are indexed by (T, max(K), O). Each set has shape (N,4,2).
        """
        K, n_ov = params.K, params.O
        T = self.__prediction_horizon
        vertices = np.empty((T, np.max(K), n_ov), dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    ps = ovehicle.pred_positions[latent_idx][:, t]
                    yaws = ovehicle.pred_yaws[latent_idx][:, t]
                    # vertices[t][latent_idx][ov_idx] = get_vertices_from_centers(
                    #     ps, yaws, ovehicle.bbox
                    # )
                    vertices[t][latent_idx][ov_idx] = util.npu.vertices_of_bboxes(
                            ps, yaws, ovehicle.bbox)

        return vertices

    def compute_overapproximations(self, params, ovehicles, vertices):
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
            Collection of A matrices of shape (T, max(K), O, L, 2).
            Where max(K) largest set of cluster.
        ndarray
            Collection of b vectors of shape (T, max(K), O, L).
            Where max(K) largest set of cluster.
        """
        K, n_ov = params.K, params.O
        T = self.__prediction_horizon
        shape = (T, np.max(K), n_ov,)
        A_union = np.empty(shape, dtype=object).tolist()
        b_union = np.empty(shape, dtype=object).tolist()
        for ov_idx, ovehicle in enumerate(ovehicles):
            for latent_idx in range(ovehicle.n_states):
                for t in range(T):
                    yaws = ovehicle.pred_yaws[latent_idx][:, t]
                    vertices_k = vertices[t][latent_idx][ov_idx]
                    mean_theta_k = np.mean(yaws)
                    A_union_k, b_union_k = compute_L4_outerapproximation(
                        mean_theta_k, vertices_k
                    )
                    A_union[t][latent_idx][ov_idx] = A_union_k
                    b_union[t][latent_idx][ov_idx] = b_union_k
        return A_union, b_union

    def do_clustering(self, frame):
        pred_result = self.do_prediction(frame)
        ovehicles = self.make_ovehicles(pred_result)
        params = self.make_local_params(frame, ovehicles)
        vertices = self.compute_vertices(params, ovehicles)
        A_union, b_union = self.compute_overapproximations(
            params, ovehicles, vertices
        )
        states = []
        bboxes = []
        for vehicle in ovehicles:
            bboxes.append(vehicle.bbox)
            if vehicle.node.id == "ego":
                vehicle = self.__ego_vehicle
            else:
                vehicle = self.__other_vehicles[int(vehicle.node.id)]
            state = carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(
                vehicle, flip_y=True
            )
            state = np.array([state[0], state[1], state[13], get_speed(vehicle)])
            states.append(state)
        states = np.stack(states)
        bboxes = np.stack(bboxes)
        self.__plot_simulation_data.states[frame] = states
        self.__plot_simulation_data.bboxes[frame] = bboxes
        self.__plot_simulation_data.vertices[frame] = vertices
        O, K = params.O, params.K
        self.__plot_simulation_data.OK_Ab_union[frame] = (O, K, A_union, b_union)

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
            debug=False,
        )

    def run_step(self, frame):
        if self.__first_frame is None:
            self.do_first_step(frame)
        self.__scene_builder.capture_trajectory(frame)
        if (frame - self.__first_frame) % self.__scene_config.record_interval == 0:
            """We only motion plan every `record_interval` frames
            (e.g. every 0.5 seconds of simulation)."""
            frame_id = int(
                (frame - self.__first_frame) / self.__scene_config.record_interval
            )
            if frame_id < self.__n_burn_interval:
                """Initially collect data without doing anything."""
                pass
            elif (frame_id - self.__n_burn_interval) % self.__step_horizon == 0:
                self.do_clustering(frame)

    def remove_scene_builder(self, first_frame):
        raise Exception(
            f"Can't remove scene builder from {util.classname(first_frame)}."
        )

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
        logging.debug(
            f"in DataCollector.parse_image() player = {self.__ego_vehicle.id} frame = {image.frame}"
        )
        self.__lidar_feeds[image.frame] = image
        if self.__scene_builder:
            self.__scene_builder.capture_lidar(image)
