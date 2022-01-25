"""Based on v1 with minor changes."""

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

from ....generate.scene import OnlineConfig
from ..util import plot_oa_simulation
from ..dynamics import (
    get_state_matrix, get_input_matrix,
    get_output_matrix, get_feedforward_matrix
)
# from ...lowlevel.v1_1 import LocalPlanner

# Local libraries
import carla
import utility as util
import carlautil
import carlautil.debug

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """

    def __init__(self, vehicle, args_lateral, args_longitudinal, offset=0, max_throttle=1., max_brake=1.,
                 max_steering=1.):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, offset, **args_lateral)

    def run_step(self, target_speed, waypoint):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoint: target location encoded as a waypoint
            :return: distance (in meters) to the waypoint
        """

        acceleration = self._lon_controller.run_step(target_speed)
        current_steering = self._lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Disable steering regulation. This should be set by midlevel controller.
        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        # if current_steering > self.past_steering + 0.1:
        #     current_steering = self.past_steering + 0.1
        # elif current_steering < self.past_steering - 0.1:
        #     current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = collections.deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, offset=0, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the vehicle invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._offset = offset
        self._e_buffer = collections.deque(maxlen=10)

    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer
        the vehicle towards a certain waypoin.

        Parameters
        ==========
        transform : carla.Transform
            Target transform.

        Returns
        =======
        float    
            steering control in the range [-1, 1] where:
                -1 maximum steering to left
                +1 maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())

    def _pid_control(self, transform, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations.

        Parameters
        ==========
        transform : carla.Transform
            Target transform.
        vehicle_transform : carla.Transform
            Current transform of the vehicle
        
        Returns
        =======
        float
            Steering control in the range [-1, 1]
        """
        # Get the ego's location and forward vector
        ego_loc = vehicle_transform.location
        v_vec = vehicle_transform.get_forward_vector()
        v_vec = np.array([v_vec.x, v_vec.y, 0.0])

        # Get the vector vehicle-target_wp
        if self._offset != 0:
            # Displace the wp to the side
            w_tran = transform
            r_vec = w_tran.get_right_vector()
            w_loc = w_tran.location + carla.Location(x=self._offset*r_vec.x,
                                                         y=self._offset*r_vec.y)
        else:
            w_loc = transform.location

        w_vec = np.array([w_loc.x - ego_loc.x,
                          w_loc.y - ego_loc.y,
                          0.0])

        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)


class LocalPlanner(object):

    def __init__(self, vehicle):
        self.__vehicle = vehicle
        self.__world = self.__vehicle.get_world()
        self.dt = self.__world.get_settings().fixed_delta_seconds

        self.args_lat = {
                'K_P': 1.0,
                'K_D': 0.0,
                'K_I': 0.0,
                'dt': self.dt}
        self.args_long = {
                'K_P': 1.0,
                'K_D': 0.0,
                'K_I': 0.0,
                'dt': self.dt}
        self.step_to_trajectory = None
        self.step_to_speed = None
        self.step = 0

    def set_plan(self, trajectory, step_period, velocity=None):
        """
        Parameters
        ==========
        trajectory : list of carla.Transform
            The trajectory to control the vehicle. The trajectory should
            not include the vehicle's current position.
        step_period : int
            The fixed number steps of between two consecutive
            points in the trajectory.
            Each step takes carla.WorldSettings.fixed_delta_seconds time.
        velocity : list of float
            Specifies velocity to control the vehicle at in m/s for each point in trajectory.
            If not provided, then infer velocity from trajectory.
        """
        n_steps = len(trajectory)
        self.step_to_trajectory = [trajectory[step] for step in range(n_steps) for _ in range(step_period)]
        self.step_to_speed      = [velocity[step] for step in range(n_steps) for _ in range(step_period)]
        self.step = 0

    @staticmethod
    def default_control():
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def run_step(self):
        logging.debug(f"In LocalPlanner.run_step() with step = {self.step}")
        if not self.step_to_trajectory:
            return self.default_control()
        elif self.step >= len(self.step_to_trajectory):
            return self.default_control()
        waypoint = self.step_to_trajectory[self.step]
        speed    = self.step_to_speed[self.step]
        args_lat = self.args_lat
        args_long = self.args_long
        pid_controller = VehiclePIDController(
                self.__vehicle,
                args_lateral=args_lat,
                args_longitudinal=args_long)
        control = pid_controller.run_step(speed, waypoint)
        self.step += 1
        return control

class MotionPlanner(object):

    def __make_global_params(self):
        """Get global parameters used across all loops"""
        params = util.AttrDict()
        # Big M variable for Slack variables in solver
        params.M = 1000
        # Control variable for solver, setting max acceleration
        params.max_a = 2.5
        # Maximum steering angle
        physics_control = self.__ego_vehicle.get_physics_control()
        wheels = physics_control.wheels
        params.max_delta = np.deg2rad(wheels[0].max_steer_angle)
        # longitudinal and lateral dimensions of car are normally 3.70 m, 1.79 m resp.
        params.bbox_lon, params.bbox_lat, _ = carlautil.actor_to_bbox_ndarray(
                self.__ego_vehicle)
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
            n_burn_interval=4,
            control_horizon=6,
            step_horizon=1,
            scene_config=OnlineConfig(),
            road_boundary_constraints=True,
            #######################
            # Logging and debugging
            log_cplex=False,
            log_agent=False,
            plot_simulation=False,
            plot_boundary=False,
            #######################
            # Planned path settings
            turn_choices=[],
            max_distance=100,
            #######################
            **kwargs):
        self.__ego_vehicle = ego_vehicle
        self.__map_reader = map_reader
        # __n_burn_interval : int
        #   Interval in prediction timesteps to skip prediction and control.
        self.__n_burn_interval = n_burn_interval
        # __control_horizon : int
        #   Number of timesteps to optimize control over.
        self.__control_horizon = control_horizon
        # __step_horizon : int
        #   Number of steps to take at each iteration of MPC.
        self.__step_horizon = step_horizon
        self.__scene_config = scene_config
        # __first_frame : int
        #   First frame in simulation. Used to find current timestep.
        self.__first_frame = None
        self.__world = self.__ego_vehicle.get_world()
        self.__timestep = self.__scene_config.record_interval \
                * self.__world.get_settings().fixed_delta_seconds
        self.__local_planner = LocalPlanner(self.__ego_vehicle)
        self.__params = self.__make_global_params()
        self.__setup_curved_road_segmented_boundary_conditions(
            turn_choices, max_distance
        )
        self.road_boundary_constraints = road_boundary_constraints
        self.log_cplex = log_cplex
        self.log_agent = log_agent
        self.plot_simulation = plot_simulation
        self.plot_boundary = plot_boundary
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

    def __plot_simulation(self):
        if len(self.__plot_simulation_data.planned_trajectories) == 0:
            return
        filename = f"agent{self.__ego_vehicle.id}_oa_simulation"
        lon, lat, _ = carlautil.actor_to_bbox_ndarray(self.__ego_vehicle)
        plot_oa_simulation(
            self.__map_reader.map_data,
            self.__plot_simulation_data.actual_trajectory,
            self.__plot_simulation_data.planned_trajectories,
            self.__plot_simulation_data.planned_controls,
            self.__plot_simulation_data.goals,
            self.__road_segs,
            [lon, lat],
            self.__step_horizon,
            filename=filename,
            road_boundary_constraints=self.road_boundary_constraints
        )

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        if self.plot_simulation:
            self.__plot_simulation()

    def get_current_steering(self):
        """Get current steering angle in radians.
        TODO: is this correct??
        https://github.com/carla-simulator/carla/issues/699
        """
        return self.__ego_vehicle.get_control().steer*self.__params.max_delta
    
    def get_current_velocity(self):
        """
        Get current velocity of vehicle in
        """
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(
                self.__ego_vehicle, flip_y=True)
        return np.sqrt(v_0_x**2 + v_0_y**2)

    def make_local_params(self, frame):
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
                self.__timestep)
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
            (0.75*v_lim*self.__timestep*self.__control_horizon) \
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

    def do_highlevel_control(self, params):
        """Decide parameters.
        """
        segs_polytopes, goal, seg_psi_bounds = self.compute_segs_polytopes_and_goal(params)

        """Apply motion planning problem"""
        n_segs = len(segs_polytopes)
        Gamma, nu, nx = params.Gamma, params.nu, params.nx
        T = self.__control_horizon
        max_a, max_delta = self.__params.max_a, self.__params.max_delta
        model = docplex.mp.model.Model(name="proposed_problem")
        min_u = np.vstack((np.zeros(T), np.full(T, -0.5*max_delta))).T.ravel()
        max_u = np.vstack((np.full(T, max_a), np.full(T, 0.5*max_delta))).T.ravel()
        u = np.array(model.continuous_var_list(nu*T, lb=min_u, ub=max_u, name='u'),
                dtype=object)
        # State variables
        X = (params.initial_rollout.local + util.obj_matmul(Gamma, u)).reshape(T + 1, nx)
        X = params.transform(X)
        X = X[1:]
        # Control variables
        U = u.reshape(T, nu)

        """Apply motion dynamics constraints"""
        model.add_constraints(self.compute_dyamics_constraints(params, X[:, 3], X[:, 4]))

        """Apply road boundary constraints"""
        if self.road_boundary_constraints:
            # Slack variables from road obstacles
            Omicron = np.array(model.binary_var_list(n_segs*T, name="omicron"),
                    dtype=object).reshape(n_segs, T)
            M_big, diag = self.__params.M, self.__params.diag
            psi_0 = params.initial_state.world[2]
            for t in range(self.__control_horizon):
                for seg_idx, (A, b) in enumerate(segs_polytopes):
                    lhs = util.obj_matmul(A, X[t, :2]) - np.array(M_big*(1 - Omicron[seg_idx, t]))
                    rhs = b# + diag
                    """Constraints on road boundaries"""
                    model.add_constraints(
                            [l <= r for (l,r) in zip(lhs, rhs)])
                    """Constraints on angle boundaries"""
                    # lhs = X[t, 2] - psi_0 - M_big*(1 - Omicron[seg_idx, t])
                    # model.add_constraint(lhs <= seg_psi_bounds[seg_idx][1])
                    # lhs = X[t, 2] - psi_0 + M_big*(1 - Omicron[seg_idx, t])
                    # model.add_constraint(lhs >= seg_psi_bounds[seg_idx][0])
                model.add_constraint(np.sum(Omicron[:, t]) >= 1)

        """Start from current vehicle position and minimize the objective"""
        w_ch_accel = 1.
        w_ch_turning = 5.
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
                goal=goal)

    def __compute_prediction_controls(self, frame):
        params = self.make_local_params(frame)
        ctrl_result = self.do_highlevel_control(params)

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
            # yaw = np.arctan2(X[t, 1] - X[t - 1, 1], X[t, 0] - X[t - 1, 0])
            yaw = np.rad2deg(util.reflect_radians_about_x_axis(yaw))
            transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation(yaw=yaw))
            trajectory.append(transform)
            velocity.append(X[t, 3])

        if self.plot_simulation:
            """Save planned trajectory for final plotting"""
            self.__plot_simulation_data.planned_trajectories[frame] = X
            self.__plot_simulation_data.planned_controls[frame] = ctrl_result.U_star
        return trajectory, velocity

    def do_first_step(self, frame):
        self.__first_frame = frame

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
                self.__plot_simulation_data.goals[frame] = self.get_goal()

        if not control:
            control = self.__local_planner.run_step()
        self.__ego_vehicle.apply_control(control)
