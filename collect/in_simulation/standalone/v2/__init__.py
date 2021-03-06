"""Curved road boundaries.
LTI bicycle model linearized around the nominal point (x, y, psi) = (0, 0, 0).

    - Uses bicycle model with steering as control.
    - Applies curved road boundaries using segmented polytopes.
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
import control
import control.matlab
import docplex.mp
import docplex.mp.model
import docplex.mp.constants

# Import modules
from ....profiling import profile
from ....generate.scene import OnlineConfig
from .util import plot_oa_simulation
from ...dynamics.bicycle_v2 import (
    get_state_matrix, get_input_matrix,
    get_output_matrix, get_feedforward_matrix
)
from ...lowlevel.v3 import VehiclePIDController

# Local libraries
import carla
import utility as util
import utility.npu
import carlautil
import carlautil.debug

class MotionPlanner(object):

    def __make_global_params(self):
        """Get scenario wide parameters used across all loops"""
        params = util.AttrDict()
        # Slack variable for solver
        params.M_big = 10_000
        # Control variable for solver, setting max/min acceleration/speed
        params.max_a = 1
        params.min_a = -7
        params.max_v = 5
        # objective : util.AttrDict
        #   Parameters in objective function. 
        params.objective = util.AttrDict(
            w_ch_accel=0., w_ch_turning=0.5,
            w_accel=0., w_turning=0.
        )
        # Maximum steering angle
        physics_control = self.__ego_vehicle.get_physics_control()
        wheels = physics_control.wheels
        params.limit_delta = np.deg2rad(wheels[0].max_steer_angle)
        # Max steering
        #   We fix max turning angle to make reasonable planned turns.
        params.max_delta = 0.5*params.limit_delta
        # longitudinal and lateral dimensions of car are normally 3.70 m, 1.79 m resp.
        params.bbox_lon, params.bbox_lat, _ = carlautil.actor_to_bbox_ndarray(
                self.__ego_vehicle)
        # Minimum distance from vehicle to avoid collision.
        #   Assumes that car is a circle.
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
        self.__road_seg_starting[2] = util.npu.reflect_radians_about_x_axis(
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
        # __road_segs : util.AttrDict
        #   Container of road segment properties.
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
            scene_config=OnlineConfig(),
            ##########################
            # Motion Planning settings
            n_burn_interval=4,
            control_horizon=6,
            step_horizon=1,
            road_boundary_constraints=True,
            angle_boundary_constraints=False,
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
        # __ego_vehicle : carla.Vehicle
        #   The vehicle to control in the simulator.
        self.__ego_vehicle = ego_vehicle
        # __map_reader : MapQuerier
        #   To query map data.
        self.__map_reader = map_reader
        # __n_burn_interval : int
        #   Interval in prediction timesteps to skip prediction and control.
        self.__n_burn_interval = n_burn_interval
        # __control_horizon : int
        #   Number of predictions steps to optimize control over.
        self.__control_horizon = control_horizon
        # __step_horizon : int
        #   Number of predictions steps to execute at each iteration of MPC.
        self.__step_horizon = step_horizon
        self.__scene_config = scene_config
        # __first_frame : int
        #   First frame in simulation. Used to find current timestep.
        self.__first_frame = None
        self.__world = self.__ego_vehicle.get_world()
        # __steptime : float
        #   Time in seconds taken to complete one step of MPC.
        self.__steptime = self.__scene_config.record_interval \
                * self.__world.get_settings().fixed_delta_seconds
        # __U_warmstarting : ndarray
        #   Controls computed from last MPC step for warmstarting.
        self.__U_warmstarting = None
        self.__local_planner = VehiclePIDController(self.__ego_vehicle)
        self.__params = self.__make_global_params()
        self.__setup_curved_road_segmented_boundary_conditions(
            turn_choices, max_distance
        )
        self.road_boundary_constraints = road_boundary_constraints
        self.angle_boundary_constraints = angle_boundary_constraints
        self.log_cplex       = log_cplex
        self.log_agent       = log_agent
        self.plot_simulation = plot_simulation
        self.plot_boundary   = plot_boundary
        if self.plot_simulation:
            self.__plot_simulation_data = util.AttrDict(
                actual_trajectory=collections.OrderedDict(),
                planned_trajectories=collections.OrderedDict(),
                planned_controls=collections.OrderedDict(),
                goals=collections.OrderedDict(),
                lowlevel=collections.OrderedDict()
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
            self.__plot_simulation_data.lowlevel,
            self.__road_segs,
            np.array([lon, lat]),
            self.__step_horizon,
            self.__steptime,
            filename=filename,
            road_boundary_constraints=self.road_boundary_constraints
        )

    def destroy(self):
        """Release all the CARLA resources used by this collector."""
        if self.plot_simulation:
            self.__plot_simulation()

    def get_current_velocity(self):
        """Get current velocity of vehicle in m/s."""
        v_0_x, v_0_y, _ = carlautil.actor_to_velocity_ndarray(
            self.__ego_vehicle, flip_y=True
        )
        return np.sqrt(v_0_x**2 + v_0_y**2)

    def make_local_params(self, frame):
        """Get the linearized bicycle model using the vehicle's
        immediate orientation."""
        params = util.AttrDict()
        params.frame = frame

        """Dynamics parameters"""
        p_0_x, p_0_y, _ = carlautil.to_location_ndarray(
            self.__ego_vehicle, flip_y=True
        )
        _, psi_0, _ = carlautil.actor_to_rotation_ndarray(
            self.__ego_vehicle, flip_y=True
        )
        v_0_mag = self.get_current_velocity()
        # initial_state - state at current frame in world/local coordinates
        #   Local coordinates has initial position and heading at 0
        x_init = np.array([0, 0, 0, v_0_mag])
        u_init = np.array([0., 0.])
        initial_state = util.AttrDict(
            world=np.array([p_0_x, p_0_y, psi_0, v_0_mag]),
            local=x_init
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
        A = get_state_matrix(x_init, u_init, l_r=0.5*bbox_lon, L=bbox_lon)
        B = get_input_matrix(x_init, u_init, l_r=0.5*bbox_lon, L=bbox_lon)
        C = get_output_matrix()
        D = get_feedforward_matrix()
        sys = control.matlab.c2d(
            control.matlab.ss(A, B, C, D),
            self.__steptime
        )
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
        """Compute the road boundary constraints and goal.

        Returns
        =======
        list of tuple of ndarray
            Polytopes with H-representation (A, b)
            where points Ax <= b iff x is in the polytope.
        ndarray
            Global (x, y) coordinates for car's destination at for the MPC step.
        list of tuple of float
            Bounds on longitudinal axis of vehicle should stay within if the
            vehicle is contined in the respective polytope.

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
            (0.75*v_lim*self.__steptime*self.__control_horizon) // segment_length + 1
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
        if self.angle_boundary_constraints:
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

    def compute_dyamics_constraints(self, params, v):
        """Set velocity magnitude constraints.
        Usually street speed limits are 30 km/h == 8.33.. m/s.
        Speed limits can be 30, 40, 60, 90 km/h

        Parameters
        ==========
        params : util.AttrDict
        v : np.array of docplex.mp.vartype.VarType
        """
        # max_v = self.__ego_vehicle.get_speed_limit() # is m/s
        max_v = self.__params.max_v
        constraints = []
        constraints.extend([ z <= max_v for z in v ])
        constraints.extend([ z >= 0     for z in v ])
        return constraints

    def compute_objective(self, X, U, goal):
        """Set the objective."""
        obj = self.__params.objective
        # final destination objective
        cost = (X[-1, 0] - goal[0])**2 + (X[-1, 1] - goal[1])**2
        # change in acceleration objective
        for u1, u2 in util.pairwise(U[:, 0]):
            _u = (u1 - u2)
            cost += obj.w_ch_accel*_u*_u
        # change in turning objective
        for u1, u2 in util.pairwise(U[:, 1]):
            _u = (u1 - u2)
            cost += obj.w_ch_turning*_u*_u
        # acceleration objective
        cost += obj.w_accel*np.sum(U[:, 0]**2)
        # turning objective
        cost += obj.w_turning*np.sum(U[:, 1]**2)
        return cost

    def do_highlevel_control(self, params):
        """Decide parameters."""
        segs_polytopes, goal, seg_psi_bounds = self.compute_segs_polytopes_and_goal(
                params)

        """Apply motion planning problem"""
        n_segs = len(segs_polytopes)
        Gamma, nu, nx = params.Gamma, params.nu, params.nx
        T = self.__control_horizon
        max_a, min_a = self.__params.max_a, self.__params.min_a
        max_delta = self.__params.max_delta
        model = docplex.mp.model.Model(name="proposed_problem")
        min_u = np.vstack((np.full(T, min_a), np.full(T, -max_delta))).T.ravel()
        max_u = np.vstack((np.full(T, max_a), np.full(T,  max_delta))).T.ravel()
        # Slack variables for control
        u = np.array(
            model.continuous_var_list(nu*T, lb=min_u, ub=max_u, name='u'), dtype=object
        )
        # State variables x, y, psi, v
        X = (params.initial_rollout.local + util.obj_matmul(Gamma, u)).reshape(T + 1, nx)
        X = params.transform(X)
        X = X[1:]
        # Control variables a, delta
        U = u.reshape(T, nu)

        """Apply motion dynamics constraints"""
        model.add_constraints(self.compute_dyamics_constraints(params, X[:, 3]))

        """Apply road boundary constraints"""
        if self.road_boundary_constraints:
            # Slack variables from road obstacles
            Omicron = np.array(
                model.binary_var_list(n_segs*T, name="omicron"), dtype=object
            ).reshape(n_segs, T)
            M_big, diag = self.__params.M_big, self.__params.diag
            psi_0 = params.initial_state.world[2]
            for t in range(self.__control_horizon):
                for seg_idx, (A, b) in enumerate(segs_polytopes):
                    lhs = util.obj_matmul(A, X[t, :2]) \
                            - np.array(M_big*(1 - Omicron[seg_idx, t]))
                    rhs = b# - diag
                    """Constraints on road boundaries"""
                    model.add_constraints(
                            [l <= r for (l,r) in zip(lhs, rhs)])
                    """Constraints on angle boundaries"""
                    if self.angle_boundary_constraints:
                        lhs = X[t, 2] - psi_0 - M_big*(1 - Omicron[seg_idx, t])
                        model.add_constraint(lhs <= seg_psi_bounds[seg_idx][1])
                        lhs = X[t, 2] - psi_0 + M_big*(1 - Omicron[seg_idx, t])
                        model.add_constraint(lhs >= seg_psi_bounds[seg_idx][0])
                model.add_constraint(np.sum(Omicron[:, t]) >= 1)
        
        """Compute and minimize objective"""
        cost = self.compute_objective(X, U, goal)
        model.minimize(cost)
        if self.road_boundary_constraints and self.__U_warmstarting is not None:
            # Warm start inputs if past iteration was run.
            warm_start = model.new_solution()
            for i, u in enumerate(self.__U_warmstarting[self.__step_horizon:]):
                warm_start.add_var_value(f"u_{2*i}", u[0])
                warm_start.add_var_value(f"u_{2*i + 1}", u[1])
            # add omicron_0 as hotfix to MIP warmstart as it needs
            # at least 1 integer value set.
            warm_start.add_var_value("omicron_0", 1)
            model.add_mip_start(
                warm_start, write_level=docplex.mp.constants.WriteLevel.AllVars
            )
        
        # model.print_information()
        # model.parameters.read.datacheck = 1
        if self.log_cplex:
            model.parameters.mip.display = 2
            s = model.solve(log_output=True)
        else:
            model.parameters.mip.display = 0
            model.solve()
        # model.print_solution()

        cost = cost.solution_value
        f = lambda x: x if isinstance(x, numbers.Number) else x.solution_value
        U_star = util.obj_vectorize(f, U)
        X_star = util.obj_vectorize(f, X)
        return util.AttrDict(cost=cost, U_star=U_star, X_star=X_star, goal=goal)

    @profile(
        sort_by="cumulative", lines_to_print=50, strip_dirs=True
    )
    def __compute_prediction_controls(self, frame):
        params = self.make_local_params(frame)
        ctrl_result = self.do_highlevel_control(params)

        """use control input next round for warm starting."""
        self.__U_warmstarting = ctrl_result.U_star

        if self.plot_simulation:
            """Save planned trajectory for final plotting"""
            X = np.concatenate((params.initial_state.world[None], ctrl_result.X_star))
            self.__plot_simulation_data.planned_trajectories[frame] = X
            self.__plot_simulation_data.planned_controls[frame] = ctrl_result.U_star

        """Get trajectory and velocity"""
        angles = util.npu.reflect_radians_about_x_axis(ctrl_result.X_star[:, 2])
        speeds = ctrl_result.X_star[:, 3]
        return speeds, angles

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
            frame_id = int((frame - self.__first_frame) \
                    / self.__scene_config.record_interval)
            if frame_id < self.__n_burn_interval:
                """Initially collect data without doing any control to the vehicle."""
                pass
            elif (frame_id - self.__n_burn_interval) % self.__step_horizon == 0:
                speeds, angles = self.__compute_prediction_controls(frame)
                self.__local_planner.set_plan(
                    speeds, angles, self.__scene_config.record_interval
                )
            if self.plot_simulation:
                """Save actual trajectory for final plotting"""
                payload = carlautil.actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(
                    self.__ego_vehicle, flip_y=True
                )
                payload = np.array([
                    payload[0], payload[1], payload[13], self.get_current_velocity(),
                ])
                self.__plot_simulation_data.actual_trajectory[frame] = payload
                self.__plot_simulation_data.goals[frame] = self.get_goal()


        if not control:
            control = self.__local_planner.step()
        self.__ego_vehicle.apply_control(control)
        if self.plot_simulation:
            payload = self.__local_planner.get_current()
            self.__plot_simulation_data.lowlevel[frame] = payload
