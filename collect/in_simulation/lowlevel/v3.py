"""PID controller based on PythonAPI.
Fixes problems with controllers v1, v1_1.
Improves upon v2 by adding piecewise linear reference trajectory.
"""
import collections
import math
import copy
import logging

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
        self.__stats = util.AttrDict(pe=0., ie=0., de=0.)

    def __store_current(self, pe, ie, de):
        """For logging"""
        self.__stats = util.AttrDict(pe=pe, ie=ie, de=de)
    
    def get_current(self):
        return copy.copy(self.__stats)

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
        # logging.info(f"step() speed current={current_speed} target={target_speed} error={error}")
        self.__error_buffer.append(error)
        if len(self.__error_buffer) >= 2:
            _de = (self.__error_buffer[-1] - self.__error_buffer[-2]) / self.__dt
            _ie = sum(self.__error_buffer) * self.__dt
        else:
            _de = 0.0
            _ie = 0.0
        self.__store_current(error, _ie, _de)
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
        self.__stats = util.AttrDict(pe=0., ie=0., de=0.)

    def __store_current(self, pe, ie, de):
        """For logging"""
        self.__stats = util.AttrDict(pe=pe, ie=ie, de=de)
    
    def get_current(self):
        return copy.copy(self.__stats)

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
        self.__store_current(error, _ie, _de)
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
            'K_P': 1.00,
            'K_I': 0.50,
            #'K_D': 0.07,
            'K_D': 0.00,
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

    def get_current(self):
        """Get reference and measurements after step has been taken step,
        and control is applied to vehicle.

        Returns
        =======
        util.AttrDict
            Contains the following at current 
            - measurement
            - reference
            - control : the last applied control.
        """
        control = self.__vehicle.get_control()
        control = util.AttrDict(
            throttle=control.throttle, brake=control.brake, steer=control.steer
        )
        error = util.AttrDict(
            speed=self.longitudinal_controller.get_current(),
            angle=self.lateral_controller.get_current()
        )
        speed = carlautil.actor_to_speed(self.__vehicle)
        _, angle, _ = carlautil.to_rotation_ndarray(self.__vehicle)
        if not self.step_to_speed \
                or self.__step_idx - 1 >= len(self.step_to_speed):
            return util.AttrDict(
                measurement=util.AttrDict(speed=speed, angle=angle),
                reference=util.AttrDict(speed=speed, angle=angle),
                error=error,
                control=control
            )
        reference_speed = self.step_to_speed[self.__step_idx - 1]
        reference_angle = self.step_to_angle[self.__step_idx - 1]
        return util.AttrDict(
            measurement=util.AttrDict(speed=speed, angle=angle),
            reference=util.AttrDict(
                speed=reference_speed,
                angle=reference_angle,
            ),
            error=error,
            control=control
        )

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
        target_speeds : iterable of float
            Target speeds in m/s for the next few consecutive steps.
        target_angles : iterable of float
            Target angles in radians for the next few consecutive steps.
            Angles should be in radians and in the Unreal Engine coordinate system.
            The iterables `target_speed` and `target_angle` should have the same length.
        step_period : int
            The fixed number of steps between two consecutive points in the trajectory.
            Each step takes `carla.WorldSettings.fixed_delta_seconds` time.
        """
        speed = carlautil.actor_to_speed(self.__vehicle)
        _, heading, _ = carlautil.to_rotation_ndarray(self.__vehicle)
        target_speeds = np.concatenate(([speed], target_speeds))
        target_angles = np.concatenate(([heading], target_angles))
        # angles are not expected to lie within (-pi, pi]. Enforce this constraint.
        target_angles = util.npu.warp_radians_neg_pi_to_pi(target_angles)
        self.step_to_speed = []
        self.step_to_angle = []
        n_steps = len(target_speeds) - 1
        for step in range(n_steps):
            nxt1 = target_angles[step+1]
            nxt2 = target_angles[step+1] + 2*np.pi
            nxt3 = target_angles[step+1] - 2*np.pi
            dif1 = abs(target_angles[step] - nxt1)
            dif2 = abs(target_angles[step] - nxt2)
            dif3 = abs(target_angles[step] - nxt3)
            if dif1 < dif2 and dif1 < dif3:
                nxt = nxt1
            elif dif2 < dif1 and dif2 < dif3:
                nxt = nxt2
            else:
                nxt = nxt3
            for substep in range(step_period):
                self.step_to_speed.append(
                    target_speeds[step] + (substep/step_period)*(target_speeds[step+1] - target_speeds[step])
                )
                self.step_to_angle.append(
                    util.npu.warp_radians_neg_pi_to_pi(
                        target_angles[step] + (substep/step_period)*(nxt - target_angles[step])
                    )
                )
        self.step_to_speed.append(target_speeds[-1])
        self.step_to_angle.append(target_angles[-1])
        self.__step_idx = 1
