import logging
import attrdict
import math
import numpy as np
import scipy
import pandas as pd
import carla

import utility as util
import carlautil
import generate.overhead as generate_overhead
import precog.utils.class_util as classu
import precog.utils.tensor_util as tensoru
import precog.utils.tfutil as tfutil

class PlayerObservation(object):

    def __get_nearby_agent_ids(self, radius=None):
        if radius is None:
            radius = self.radius
        player_location = carlautil.transform_to_location_ndarray(self.player_transform)
        """When there are road above / below ego vehicle,
        then we don't want to include them."""
        upperbound_z = 4
        lowerbound_z = -4
        other_ids =  util.map_to_ndarray(lambda v: v.id, self.other_vehicles)
        other_locations = util.map_to_ndarray(
                lambda v: carlautil.transform_to_location_ndarray(v.get_transform()),
                self.other_vehicles)
        distances = np.linalg.norm(other_locations - player_location, axis=1)
        z_displacements = other_locations[:, -1] - player_location[-1]
        df = pd.DataFrame({
                'ids': other_ids,
                'distances': distances,
                'z_displacements': z_displacements })
        df = df[df['distances'] < radius]
        df = df[df['z_displacements'].between(lowerbound_z, upperbound_z, inclusive=False)]
        df.sort_values('distances', inplace=True)
        return df['ids'].to_numpy()

    def old_get_nearby_agent_ids(self, radius=None):
        """Gets all IDs of other vehicles that are radius away from ego vehicle and
        returns IDs sorted by nearest to player first.
        
        TODO: don't include other vehicles below / above th ego vehicle
        TODO: delete
        """
        if radius is None:
            radius = self.radius
        player_location = carlautil.transform_to_location_ndarray(self.player_transform)
        other_ids =  util.map_to_ndarray(lambda v: v.id, self.other_vehicles)
        other_locations = util.map_to_ndarray(
                lambda v: carlautil.transform_to_location_ndarray(v.get_transform()),
                self.other_vehicles)
        distances = np.linalg.norm(other_locations - player_location, axis=1)
        df = pd.DataFrame({ 'ids': other_ids, 'distances': distances })
        df = df[df['distances'] < radius]
        df.sort_values('distances', inplace=True)
        return df['ids'].to_numpy()

    @classu.member_initialize
    def __init__(self, frame, phi, world, other_vehicles,
            player_transforms, others_transforms, player_bbox,
            other_id_ordering=None, radius=200):
        """
        1. generates a history of player and other vehicle position coordinates
           of size len(player_transforms)

        Parameters
        ----------
        frame : int
            Frame ID of observation
        world : carla.World
        other_vehicles : list of carla.Vehicle
        player_transform : collections.deque of carla.Transform
            Collection of transforms of player
            Ordered by timestep where last in deque is current timestep
        others_transforms : collections.deque of (dict of int : carla.Trajectory)
            Collection of transforms of other vehicles by vehicle ID
            Ordered by timestep where last in deque is current timestep
        A : int
            Number of vehicles in observation, including ego vehicle. Must be A > 1.
        other_id_ordering : list of int
            IDs of other (not player) vehicles (not walkers).
            All IDs in other_id_ordering belong to some vehicle ID
        radius : int
        """
        _, _, self.T_past, _ = tensoru.shape(self.phi.S_past_world_frame)
        self.B, self.A, self.T, self.D = tensoru.shape(self.phi.S_future_world_frame)
        self.B, self.H, self.W, self.C = tensoru.shape(self.phi.overhead_features)
        assert(len(player_transforms) == len(others_transforms))
        assert(self.A > 0)

        # player_transform : carla.Transform
        # transform of current player 
        self.player_transform = player_transforms[-1]

        # player_positions_world : ndarray of shape (len(player_transforms), 3)
        self.player_positions_world = carlautil.transforms_to_location_ndarray(
                self.player_transforms)
        # player_positions_local : ndarray of shape (len(player_transforms), 3)
        self.player_positions_local = self.player_positions_world \
                - carlautil.transform_to_location_ndarray(self.player_transform)
        # player_yaw : float
        self.player_yaw = carlautil.transform_to_yaw(self.player_transform)

        if self.other_id_ordering is None:
            # get list of A agents within radius close to us
            # note that other_id_ordering may have size smaller than A-1
            ids = self.__get_nearby_agent_ids()
            self.other_id_ordering = ids[:self.A - 1]

        if self.other_id_ordering.shape != (0,):
            others_positions_list = [None] * len(self.others_transforms)
            for idx, others_transform in enumerate(self.others_transforms):
                others_positions_list[idx] = util.map_to_list(
                        lambda aid: carlautil.transform_to_location_ndarray(others_transform[aid]),
                        self.other_id_ordering)

            # agent_positions_world : ndarray of shape (A-1, len(self.others_transforms), 3)
            self.agent_positions_world = np.array(others_positions_list).transpose(1, 0, 2)

            others_transform = self.others_transforms[-1]
            # agent_yaws : ndarray of shape (A-1,)
            self.agent_yaws = util.map_to_ndarray(
                    lambda aid: carlautil.transform_to_yaw(others_transform[aid]),
                    self.other_id_ordering)

            self.n_missing = max(self.A - 1 - self.agent_positions_world.shape[0], 0)
            self.has_other_agents = True
        else:
            self.agent_positions_world = np.array([])
            self.agent_yaws = np.array([])
            self.n_missing = self.A - 1
            self.has_other_agents = False

        if self.n_missing > 0:
            faraway_position = carlautil.transform_to_location_ndarray(self.player_transform) \
                    + np.array([0, 300, 0])
            faraway_tile = np.tile(
                    faraway_position, (self.n_missing, len(self.others_transforms), 1))
            if self.n_missing == self.A - 1:
                self.agent_positions_world = faraway_tile
                self.agent_yaws = np.zeros(self.A-1)
            else:
                self.agent_positions_world = np.concatenate(
                        (self.agent_positions_world, faraway_tile), axis=0)
                self.agent_yaws = np.concatenate(
                        (self.agent_yaws, np.zeros(self.n_missing)), axis=0)
        
        # agent_positions_local : ndarray of shape (A-1, len(self.others_transforms), 3)
        self.agent_positions_local = self.agent_positions_world \
                - carlautil.transform_to_location_ndarray(self.player_transform)
    
    @property
    def n_present(self):
        return self.A - self.n_missing

    def copy_with_new_ordering(self, other_id_ordering):
        return PlayerObservation(self.frame, self.phi, self.world,
                self.other_vehicles, self.player_transforms,
                self.others_transforms, self.player_bbox,
                other_id_ordering=other_id_ordering, radius=self.radius)


class DrivingSample(object):
    """Represents a dataset sample. Responsible for further preprocessing of data.
    Used for data augmentation."""

    @classu.member_initialize
    def __init__(self, episode, frame, lidar_params, sample_labels,
            save_directory, sample_name, lidar_measurement, player_bbox,
            player_past, agent_pasts, player_future, agent_futures,
            player_yaw, agent_yaws,
            should_augment=False, n_augments=1):
        """
        Parameters
        ----------
        sample_name : str
            File name without extension to name sample.
        lidar_measurement : carla.LidarMeasurement or carla.SemanticLidarMeasurement
            LIDAR capture to add to sample.
        player_past : np.array
            Player past trajectory; np.array of shape (T_past, 3).
        agent_pasts : np.array
            Agent pasts trajectory; np.array of shape (A-1, T_past, 3).
        player_future
        agent_futures

        Add more augmentations, add dropout and noise to semantic LIDAR.
        """
        # lidar_points : np.array
        #     Has shape (number of points, 3) if using carla.LidarMeasurement.
        # lidar_point_labels : np.array
        #     Labels for the LIDAR points. Each row should correspond to the same
        #     row in lidar_points
        self.lidar_points, self.lidar_point_labels \
                = generate_overhead.get_normalized_sensor_data(lidar_measurement)
        # uncomment to add rudimentary gaussian noise to lidar points
        # if isinstance(lidar_measurement, carla.SemanticLidarMeasurement):
        #     """Semantic LIDAR does not come with dropout or noise"""
        #     self.lidar_points += np.random.uniform(
        #             -0.05, 0.05, size=self.lidar_points.shape)

    def __save_sample(self, sample_name, lidar_points, player_past,
            agent_pasts, player_future, agent_futures,
            player_yaw, agent_yaws):
        overhead_features = generate_overhead.build_BEV(
                lidar_points, self.lidar_params, self.player_bbox,
                lidar_point_labels=self.lidar_point_labels)
        player_past = player_past[:, :2]
        agent_pasts = agent_pasts[:, :, :2]
        player_future = player_future[:, :2]
        agent_futures = agent_futures[:, :, :2]

        datum = {}
        datum['episode'] = self.episode
        datum['frame'] = self.frame
        datum['lidar_params'] = vars(self.lidar_params)
        datum['player_past'] = player_past
        datum['agent_pasts'] = agent_pasts
        datum['overhead_features'] = overhead_features
        datum['player_future'] = player_future
        datum['agent_futures'] = agent_futures
        datum['player_yaw'] = player_yaw
        datum['agent_yaws'] = agent_yaws
        datum['labels'] = vars(self.sample_labels)
        util.save_datum(datum, self.save_directory, sample_name)

    def __save_orient_to_ego(self):
        R = scipy.spatial.transform.Rotation

        """Random rotation about origin."""
        # why am I using revrotmat instead of rotmat
        angle = math.radians(self.player_yaw)
        rotmat = R.from_rotvec(np.array([0, 0, -1]) * angle).as_matrix()
        revrotmat = R.from_rotvec(np.array([0, 0, 1]) * angle).as_matrix()
        lidar_points = (revrotmat @ self.lidar_points.T).T
        player_past = (rotmat @ self.player_past.T).T
        n_oagents = self.agent_pasts.shape[0]
        agent_pasts = np.reshape(self.agent_pasts, (-1, 3))
        agent_pasts = (rotmat @ agent_pasts.T).T
        agent_pasts = np.reshape(agent_pasts, (n_oagents, -1, 3))
        player_future = (rotmat @ self.player_future.T).T
        n_oagents = self.agent_futures.shape[0]
        agent_futures = np.reshape(self.agent_futures, (-1, 3))
        agent_futures = (rotmat @ agent_futures.T).T
        agent_futures = np.reshape(agent_futures, (n_oagents, -1, 3))
        player_yaw = (self.player_yaw - math.degrees(angle)) % 360.
        agent_yaws = (self.agent_yaws - math.degrees(angle)) % 360.

        self.__save_sample(self.sample_name, lidar_points, player_past,
                agent_pasts, player_future, agent_futures,
                player_yaw, agent_yaws)

    def __generate_augmentations(self):
        R = scipy.spatial.transform.Rotation
        n_augments = np.random.randint(1, self.n_augments+1)

        for idx in range(n_augments):
            sample_name = f"{self.sample_name}_aug{idx}"

            """Random rotation about origin."""
            # why am I using revrotmat instead of rotmat
            angle = (np.random.sample()*2 - 1)*np.pi
            rotmat = R.from_rotvec(np.array([0, 0, -1]) * angle).as_matrix()
            revrotmat = R.from_rotvec(np.array([0, 0, 1]) * angle).as_matrix()
            lidar_points = (revrotmat @ self.lidar_points.T).T
            player_past = (rotmat @ self.player_past.T).T
            n_oagents = self.agent_pasts.shape[0]
            agent_pasts = np.reshape(self.agent_pasts, (-1, 3))
            agent_pasts = (rotmat @ agent_pasts.T).T
            agent_pasts = np.reshape(agent_pasts, (n_oagents, -1, 3))
            player_future = (rotmat @ self.player_future.T).T
            n_oagents = self.agent_futures.shape[0]
            agent_futures = np.reshape(self.agent_futures, (-1, 3))
            agent_futures = (rotmat @ agent_futures.T).T
            agent_futures = np.reshape(agent_futures, (n_oagents, -1, 3))
            player_yaw = (self.player_yaw + math.degrees(angle)) % 360.
            agent_yaws = (self.agent_yaws + math.degrees(angle)) % 360.

            """Shift points about (x,y)-plane."""
            ## can't get this to work right now.
            # unit = np.array([1,0,0])
            # angle = (np.random.sample()*2 - 1)*np.pi
            # mag = np.random.sample()*10
            # rotmat = R.from_rotvec(np.array([0, 0, -1]) * angle).as_matrix()
            # revrotmat = R.from_rotvec(np.array([0, 0, 1]) * angle).as_matrix()
            # shift = rotmat @ (mag * unit)
            # revshift = revrotmat @ (mag * unit)
            # lidar_points = lidar_points + revshift
            # player_past = player_past + shift
            # agent_pasts = agent_pasts + shift
            # player_future = player_future + shift
            # agent_futures = agent_futures + shift

            self.__save_sample(sample_name, lidar_points, player_past,
                    agent_pasts, player_future, agent_futures,
                    player_yaw, agent_yaws)

    def save(self):
        if self.should_augment:
            self.__generate_augmentations()
        else:
            self.__save_orient_to_ego()
            # self.__save_sample(self.sample_name, self.lidar_points, self.player_past,
            #         self.agent_pasts, self.player_future, self.agent_futures,
            #         self.player_yaw, self.agent_yaws)


class StreamingGenerator(object):
    """Generates driving scenario samples by combining streams of trajectories,
    and LIDAR point clouds."""

    @classu.member_initialize
    def __init__(self, phi, should_augment=False, n_augments=1):
        """
        Parameters
        ----------
        phi : ESPPhiData
        should_augment : bool
        """
        _, _, self.T_past, _ = tensoru.shape(self.phi.S_past_world_frame)
        self.B, self.A, self.T, self.D = tensoru.shape(self.phi.S_future_world_frame)
        self.B, self.H, self.W, self.C = tensoru.shape(self.phi.overhead_features)
        
    def add_feed(self, frame, observation, trajectory_feeds):
        """Updates trajectory feed at frame.
        based on StreamingCARLALoader.populate_phi_feeds

        Parameters
        ----------
        observation : PlayerObservation
        trajectory_feeds : collection.OrderedDict
        """
        player_past = observation.player_positions_local[-self.T_past:, :3]
        agent_pasts = observation.agent_positions_local[:, -self.T_past:, :3]
        player_yaw = observation.player_yaw
        agent_yaws = observation.agent_yaws

        feed_dict = attrdict.AttrDict({
                'player_past': player_past,
                'agent_pasts': agent_pasts,
                'player_yaw': player_yaw,
                'agent_yaws': agent_yaws})

        # not sure how agent_presence is being used to train PRECOG
        # agent_presence = np.ones(shape=tensoru.shape(phi.agent_presence), dtype=np.float32)
        # TODO: set yaws
        # yaws = np.tile(np.asarray(observation.yaws_local[:A])[None], (B, 1))
        # TODO: set traffic light
        # light_string_batch = np.tile(np.asarray(light_string), (B,))
        # feed_dict[phi.light_strings] = light_string_batch
        # feed_dict.validate()
        trajectory_feeds[frame] = (
                observation.player_transform,
                observation.other_id_ordering,
                feed_dict)

    def save_dataset_sample(self, frame, episode,
            observation, trajectory_feeds, lidar_feeds,
            player_bbox, lidar_sensor, lidar_params,
            save_directory, make_sample_name,
            sample_labels):
        """

        The original PRECOG dataset has these relevant keys:
        - episode : int
        - frame : int
        - lidar_params : dict with keys ['hist_max_per_pixel',
            'val_obstacle', 'meters_max', 'pixels_per_meter']
        - player_future : matrix of shape (20, 3)
        - agent_futures : matrix of shape (4, 20, 3)
        - player_past : matrix of shape (10, 3)
        - agent_pasts : matrix of shape (4, 10, 3)
        - overhead_features : matrix shape (200, 200, 4)

        Parameters
        ----------
        player_bbox : carla.BoundingBox
            Player's bounding box used to get vehicle dimensions.
        lidar_sensor : carla.Sensor
        lidar_params : LidarParams
            Lidar parameters
        save_directory : str
            Directory to save to.
        make_sample_name : lambda frame
            Function to generate name for sample
        sample_labels : SampleLabelMap
        """
        earlier_frame = frame - self.T
        player_transform, other_id_ordering, \
                feed_dict = trajectory_feeds[earlier_frame]
        ## TODO: Maybe add some check to not save samples without other agents. 
        # if not observation.has_other_agents:
        #     logging.debug("Could not obtain other agents for this sample. Skipping.")
        #     return
        observation = observation.copy_with_new_ordering(other_id_ordering)
        lidar_measurement = lidar_feeds[earlier_frame]
        player_past = feed_dict.player_past
        agent_pasts = feed_dict.agent_pasts
        player_yaw = feed_dict.player_yaw
        agent_yaws = feed_dict.agent_yaws
        player_future = observation.player_positions_world[1:self.T+1, :3] \
                - carlautil.transform_to_location_ndarray(player_transform)
        agent_futures = observation.agent_positions_world[:, 1:self.T+1, :3] \
                - carlautil.transform_to_location_ndarray(player_transform)
        sample_labels.n_present = observation.n_present

        sample = DrivingSample(
                episode=episode,
                frame=frame,
                lidar_params=lidar_params,
                sample_labels=sample_labels,
                save_directory=save_directory,
                sample_name=make_sample_name(earlier_frame),
                lidar_measurement=lidar_measurement,
                player_bbox=player_bbox,
                player_past=player_past,
                agent_pasts=agent_pasts,
                player_future=player_future,
                agent_futures=agent_futures,
                player_yaw=player_yaw,
                agent_yaws=agent_yaws,
                should_augment=self.should_augment,
                n_augments=self.n_augments)
        sample.save()
