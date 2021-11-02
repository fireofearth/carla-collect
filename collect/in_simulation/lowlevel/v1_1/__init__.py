
"""This module contains PID controllers to
perform lateral and longitudinal control."""

from collections import deque
import logging
import math
import numpy as np
import carla

import utility as util
import carlautil
import carlautil.debug

def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

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
        self._error_buffer = deque(maxlen=10)

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
        self._e_buffer = deque(maxlen=10)

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

        self.args_lat_hw_dict = {
                'K_P': 0.75,
                'K_D': 0.02,
                'K_I': 0.4,
                'dt': self.dt}
        self.args_long_hw_dict = {
                'K_P': 0.37,
                'K_D': 0.024,
                'K_I': 0.032,
                'dt': self.dt}

        self.args_lat_city_dict = {
                'K_P': 0.58,
                'K_D': 0.02,
                'K_I': 0.5,
                'dt': self.dt}
        self.args_long_city_dict = {
                'K_P': 0.15,
                'K_D': 0.05,
                'K_I': 0.07,
                'dt': self.dt}

        ## modify/test PID parameters
        # self.args_lat_city_dict = {
        #         'K_P': 0.50,
        #         'K_D': 0.02,
        #         'K_I': 0.50,
        #         'dt': self.dt}
        # self.args_long_city_dict = {
        #         'K_P': 0.50,
        #         'K_D': 0.05,
        #         'K_I': 0.20,
        #         'dt': self.dt}
        ##
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
        def get_velocity_ms(t1, t2):
            """Get velocity in m/s"""
            l1 = t1.location
            l2 = t2.location
            d = np.sqrt((l2.x - l1.x)**2 + (l2.y - l1.y)**2)
            return d / (self.dt * float(step_period))

        n_steps = len(trajectory)
        self.step_to_trajectory = [trajectory[step] for step in range(n_steps) for _ in range(step_period)]
        if velocity is None:
            velocities = util.pairwise_do(get_velocity_ms, [self.__vehicle.get_transform()] + trajectory)
            self.step_to_speed = [velocities[step] for step in range(n_steps) for _ in range(step_period)]
        else:
            self.step_to_speed = [velocity[step] for step in range(n_steps) for _ in range(step_period)]
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

    @staticmethod
    def ms_to_mkh(v):
        return 3.6 * v

    def run_step(self):
        logging.debug(f"In LocalPlanner.run_step() with step = {self.step}")
        if not self.step_to_trajectory:
            return self.default_control()
        elif self.step >= len(self.step_to_trajectory):
            return self.default_control()
        waypoint = self.step_to_trajectory[self.step]
        speed    = self.ms_to_mkh(self.step_to_speed[self.step])
        args_lat = self.args_lat_city_dict
        args_long = self.args_long_city_dict
        pid_controller = VehiclePIDController(
                self.__vehicle,
                args_lateral=args_lat,
                args_longitudinal=args_long)
        control = pid_controller.run_step(speed, waypoint)
        self.step += 1
        return control

