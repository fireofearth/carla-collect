"""PID controller based on PythonAPI.
Fixes problems with controllers v1, v1_1.
Improves upon v2 by adding piecewise linear reference trajectory.
"""

import collections
import math
import copy

import numpy as np

import utility as util
import utility.npu
import carlautil
import carla

class PIDLongitudinalController(object):
    """Implements longitudinal control using a PID.
    Based on PythonAPI provided by CARLA"""

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """Constructor method.

        Parameters
        ==========
        vehicle : carla.Vehicle
            Actor to apply to local planner logic onto
        K_P : float
            Proportional term
        K_I : float
            Integral term
        K_D : float
            Differential term
        dt : float
            Time differential in seconds
        """
        self.__vehicle = vehicle
        self.__k_p = K_P
        self.__k_d = K_D
        self.__k_i = K_I
        self.__dt = dt
        self.__clip = util.Clip(low=-1, high=1)
        self.__error_buffer = collections.deque(maxlen=8_000)

    def step(self, target_speed):
        """Execute one step of longitudinal control to reach a given target speed.
        NOTE: this method suggests that with K_P = 1, K_I = K_D = 0,
              then use full throttle when error is over 1 Km/h, or 3.6 m/s.

        Parameters
        ==========
        target_speed : float
            Target speed in m/s.
        
        Returns
        =======
        float
            Values [-1,1] for throttle or break control
        """
        current_speed = carlautil.actor_to_speed(self.__vehicle)
        error = target_speed - current_speed
        self.__error_buffer.append(error)
        if len(self.__error_buffer) >= 2:
            _de = (self.__error_buffer[-1] - self.__error_buffer[-2]) / self.__dt
            _ie = sum(self.__error_buffer) * self.__dt
        else:
            _de = 0.0
            _ie = 0.0
        ctrl_input = (self.__k_p * error) + (self.__k_d * _de) + (self.__k_i * _ie)
        return self.__clip(ctrl_input)


class PIDLateralController(object):
    """Implements lateral control using a PID.
    Based on PythonAPI provided by CARLA"""

    def __init__(self, vehicle, max_steering=1.0, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """Constructor method.

        Parameters
        ==========
        vehicle : carla.Vehicle
            Actor to apply to local planner logic onto
        max_steering : float
            The maximum steering angle in radians. For CARLA version 0.9 the Audi A2
            can steer a maximum of 57.5 degrees, 1.00 radians (average of two wheels).
        K_P : float
            Proportional term
        K_I : float
            Integral term
        K_D : float
            Differential term
        dt : float
            Time differential in seconds
        """
        self.__vehicle = vehicle
        self.__max_steering = max_steering
        self.__k_p = K_P
        self.__k_i = K_I
        self.__k_d = K_D
        self.__dt = dt
        self.__clip = util.Clip(low=-1, high=1)
        self.__error_buffer = collections.deque(maxlen=8_000)


    def step(self, target_angle):
        """Execute one step of lateral control reach a certain target vehicle angle.
        NOTE: target and measurements are of vehicle longitudinal angle.

        Parameters
        ==========
        target_angle : float
            Target vehicle longitudinal angle in radians.

        Returns
        =======
        float
            Steering control in the range [-1, 1] where
            -1 maximum steering to left and
            +1 maximum steering to right
        """
        _, current_angle, _ = carlautil.to_rotation_ndarray(self.__vehicle)
        current_angle = util.npu.warp_radians_about_center(current_angle, target_angle)
        error = target_angle - current_angle
        self.__error_buffer.append(error)
        if len(self.__error_buffer) >= 2:
            _de = (self.__error_buffer[-1] - self.__error_buffer[-2]) / self.__dt
            _ie = sum(self.__error_buffer) * self.__dt
        else:
            _de = 0.0
            _ie = 0.0
        ctrl_input = (self.__k_p * error) + (self.__k_d * _de) + (self.__k_i * _ie)
        ctrl_input = ctrl_input / self.__max_steering
        return self.__clip(ctrl_input)

class VehiclePIDController(object):
    """VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to compute the low level control inputs for a vehicle.
    """

    def __init__(self, vehicle, max_steering=1.0):
        """Constructor.

        Parameters
        ==========
        vehicle : carla.Vehicle
            Actor to apply to local planner logic onto
        max_steering : float
            The maximum steering angle in radians. For CARLA version 0.9 the Audi A2
            can steer a maximum of 57.5 degrees, 1.00 radians (average of two wheels).
        """
        self.__vehicle = vehicle
        self.__world = self.__vehicle.get_world()
        self.__dt = self.__world.get_settings().fixed_delta_seconds
        self.__args_lateral_dict = {
            'K_P': 5.70,
            'K_I': 0.00,
            'K_D': 0.00,
            'dt': self.__dt}

        self.__args_longitudinal_dict = {
            'K_P': 0.90,
            'K_I': 0.28,
            'K_D': 0.07,
            'dt': self.__dt}

        self.__max_steering = max_steering
        self.longitudinal_controller = PIDLongitudinalController(
                self.__vehicle, **self.__args_longitudinal_dict)
        self.lateral_controller      = PIDLateralController(
                self.__vehicle, max_steering=self.__max_steering,
                **self.__args_lateral_dict)
        self.step_to_speed = None
        self.step_to_angle = None
        self.__step_idx = 0

    @staticmethod
    def default_control():
        return carlautil.create_gear_control()

    def step(self):
        """Take a step in control.
        
        Parameters
        ==========
        target_speed : float
            Target speed in m/s.
        target_angle : float
            Target angle in radians.
        
        Returns
        =======
        carla.VehicleControl
            The control for vehicle to reach target.
        """
        if not self.step_to_speed:
            return self.default_control()
        elif self.__step_idx >= len(self.step_to_speed):
            return self.default_control()
        target_speed = self.step_to_speed[self.__step_idx]
        target_angle = self.step_to_angle[self.__step_idx]
        throttle_break = self.longitudinal_controller.step(target_speed)
        _steering = self.lateral_controller.step(target_angle)
        _throttle = max(0, throttle_break)
        _brake = abs(min(0, throttle_break))
        control = carlautil.create_gear_control(
            throttle=_throttle,
            steer=_steering,
            brake=_brake
        )
        self.__step_idx += 1
        return control

    def set_plan(self, target_speeds, target_angles, step_period):
        """Given a trajectory consisting of heading angle and
        velocities, use lateral and longitudinal PID controllers
        to actuate the vehicle.

        Parameters
        ==========
        target_speeds : list of float
            Target speeds in m/s for the next few consecutive steps.
        target_angles : list of float
            Target angles in radians for the next few consecutive steps.
            The lists `target_speed` and `target_angle` should have the same length.
        step_period : int
            The fixed number of steps between two consecutive points in the trajectory.
            Each step takes `carla.WorldSettings.fixed_delta_seconds` time.
        """
        speed = carlautil.actor_to_speed(self.__vehicle)
        _, heading, _ = carlautil.to_rotation_ndarray(self.__vehicle, flip_y=False)
        target_speeds = [speed] + target_speeds
        target_angles = [heading] + target_angles
        self.step_to_speed = []
        self.step_to_angle = []
        n_steps = len(target_speeds) - 1
        for step in range(n_steps):
            for substep in range(step_period):
                self.step_to_speed.append(
                    target_speeds[step] + (substep/step_period)*(target_speeds[step+1] - target_speeds[step])
                )
                self.step_to_angle.append(
                    target_angles[step] + (substep/step_period)*(target_angles[step+1] - target_angles[step])
                )
        self.step_to_speed.append(target_speeds[-1])
        self.step_to_angle.append(target_angles[-1])
        self.__step_idx = 1
