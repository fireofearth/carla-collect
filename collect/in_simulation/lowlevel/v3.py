"""PID controller based on PythonAPI.
Fixes problems with controllers v1, v1_1, and uses
actual steering angle to compute errors."""

import collections
import math

import utility as util
import carlautil
import carla

def get_speed(vehicle):
    """Compute speed of a vehicle in Km/h.
    Copied from PythonAPI"""
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

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
        self.__error_buffer = collections.deque(maxlen=10)

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
        current_speed = get_speed(self.__vehicle)
        target_speed = 3.6*target_speed
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
        self.__error_buffer = collections.deque(maxlen=10)


    def step(self, target_angle):
        """Execute one step of lateral control reach a certain target steering angle.
        NOTE: this method suggests that with K_P = 1, K_I = K_D = 0, then adjust
              steering is equivalent to the normalized error.

        Parameters
        ==========
        target_angle : float
            Target angle in radians.

        Returns
        =======
        float
            Steering control in the range [-1, 1] where
            -1 maximum steering to left and
            +1 maximum steering to right
        """
        current_angle = carlautil.get_steering_angle(self.__vehicle)
        error = math.radians(target_angle - current_angle)
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
        
        """These settings are taken from PythonAPI for CARLA 0.9.12.
        It expects timesteps of 0.05 seconds."""
        self.__args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': self.__dt}
        self.__args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self.__dt}

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
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

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
        control = carla.VehicleControl(
            throttle=_throttle,
            steer=_steering,
            brake=_brake
        )
        self.step += 1
        return control


    def set_plan(self, target_speeds, traget_angles, step_period):
        """Given a trajectory consisting of steering angles and
        velocities, use lateral and longitudinal PID controllers
        to1

        Parameters
        ==========
        target_speed : list of float
            Target speeds in m/s.
        target_angle : list of float
            Target angles in radians.
            The lists `target_speed` and `target_angle` should have the same length.
        step_period : int
            The fixed number of steps between two consecutive points in the trajectory.
            Each step takes `carla.WorldSettings.fixed_delta_seconds` time.
        """
        n_steps = len(target_speeds)
        self.step_to_speed = [
            target_speeds[step] for step in range(n_steps) for _ in range(step_period)
        ]
        self.step_to_angle = [
            traget_angles[step] for step in range(n_steps) for _ in range(step_period)
        ]
        self.__step_idx = 0
