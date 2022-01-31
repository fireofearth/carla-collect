import math

import numpy as np
import scipy
import scipy.linalg
import control

sin = math.sin
cos = math.cos
atan = math.atan
tan = math.tan

def get_beta(delta, l_r=0.5, L=1.0):
    if l_r == L:
        return delta
    else:
        return atan((l_r / L)*tan(delta))

def get_dbeta_ddelta(delta, l_r=0.5, L=1.0):
    if l_r == L:
        return 1
    else:
        tan2 = tan(delta)**2
        return (1 + tan2) / ((L / l_r) + (l_r / L)*tan2)

def bicycle_kinematics(t, x, u, params):
    l_r   = params.get('l_r', 0.5)
    L     = params.get('L', 1)
    delta = params.get('delta', 0)
    psi   = x[2]
    v     = x[3]
    u_1   = u[0] # acceleration
    u_2   = u[1] # steering
    beta  = get_beta(delta, l_r=l_r, L=L)
    return np.array([
        v*cos(psi + beta), # dot x 
        v*sin(psi + beta), # dot y
        (v / L)*cos(beta)*tan(u_2), # dot psi
        u_1, # dot v
    ])

def compute_nonlinear_dynamical_states(initial_state, T, Ts, U, l_r=0.5, L=1.0):
    """Compute non-linear dynamical states from bicycle kinematic model.

    Parameters
    ==========
    initial_state : ndarray
        Initial state [x_0, y_0, psi_0, v_0].
    T : int
        Timesteps.
    Ts : float
        Step size in seconds.
    U : ndarray
        Control inputs of shape (T, 2) [u_1, u_2]
    l_r : float
        Length of hind wheel to bicycle center of gravity.
    L : float
        Length of bicycle longitudinal dimension from hind to front wheel.
    
    Returns
    =======
    ndarray
        Trajectory from control of shape (T + 1, 4) with rows [x, y, psi, v].
    """
    timestamps = np.linspace(0, Ts*T, T + 1)
    mock_inputs = np.concatenate((U, np.array([0, 0])[None]), axis=0).T
    io_bicycle_kinematics = control.NonlinearIOSystem(
                bicycle_kinematics, None,
                inputs=('u_1', 'u_2'),
                outputs=('x', 'y', 'psi', 'v'),
                states=('x', 'y', 'psi', 'v'),
                params={'l_r': l_r, 'L': L},
                name='bicycle_kinematics')
    _, states = control.input_output_response(
            io_bicycle_kinematics, timestamps, mock_inputs, initial_state)
    return states.T

def compute_nominal_trajectory(initial_state, T, Ts, u, l_r=0.5, L=1.0):
    """Compute non-linear dynamical states from bicycle kinematic model.

    Parameters
    ==========
    initial_state : ndarray
        Initial state [x_0, y_0, psi_0, v_0].
    T : int
        Timesteps.
    Ts : float
        Step size in seconds.
    u : ndarray
        Initial ontrol input [u_1, u_2]
    l_r : float
        Length of hind wheel to bicycle center of gravity.
    L : float
        Length of bicycle longitudinal dimension from hind to front wheel.
    
    Returns
    =======
    ndarray
        Trajectory from control of shape (T + 1, 4) with rows [x, y, psi, v].
    """
    U_bar = np.repeat(u[None], T, axis=0)
    X_bar = compute_nonlinear_dynamical_states(
        initial_state, T, Ts, U_bar, l_r=0.5, L=1.0
    )
    return X_bar, U_bar


def get_state_matrix(z, u, l_r=0.5, L=1.0):
    x, y, psi, v = z
    a, delta = u
    beta   = get_beta(delta, l_r=l_r, L=L)
    df3_dv = (1/L)*cos(beta)*tan(delta)
    return np.array([
        # x, y, psi, v
        [0, 0, -v*sin(psi + beta), cos(psi + beta)],
        [0, 0,  v*cos(psi + beta), sin(psi + beta)],
        [0, 0,                  0,          df3_dv],
        [0, 0,                  0,               0],
    ])

def get_input_matrix(z, u, l_r=0.5, L=1.0):
    x, y, psi, v = z
    a, delta = u
    beta   = get_beta(delta, l_r=l_r, L=L)
    dbeta  = get_dbeta_ddelta(delta, l_r=l_r, L=L)
    tan2   = tan(delta)**2
    return np.array([
        # a, delta
        [0, -v*sin(psi + beta)*dbeta],
        [0,  v*cos(psi + beta)*dbeta],
        [0, (v/L)*(cos(beta)*(1 + tan2) - sin(beta)*tan(delta)*dbeta)],
        [1,  0],
    ])

def get_output_matrix():
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

def get_feedforward_matrix():
    return np.array([
        [0, 0],
        [0, 0],
    ])

class VehicleModel(object):
    """Used to generate trajectories using non-linear bicycle kinematics model."""

    def __init__(self, T, Ts, l_r=0.5, L=1.0):
        """Constructor.

        Parameters
        ==========
        T : int
            Timesteps.
        Ts : float
            Step size in seconds.
        l_r : float
            Length of hind wheel to bicycle center of gravity.
        L : float
            Length of bicycle longitudinal dimension from hind to front wheel.
        """
        self.T = T
        self.Ts = Ts
        self.timestamps = np.linspace(0, Ts*T, T + 1)
        self.l_r = l_r
        self.L = L
        self.io_dynamical_system = control.NonlinearIOSystem(
            bicycle_kinematics, None,
            inputs=('u_1', 'u_2'),
            outputs=('x', 'y', 'psi', 'v'),
            states=('x', 'y', 'psi', 'v'),
            params={'l_r': l_r, 'L': L},
            name='bicycle_kinematics'
        )

    def states_from_control(self, x_init, U):
        """Compute non-linear states from vehicle model
        given initial states and control variables over time.

        Parameters
        ==========
        x_init : ndarray
            Initial state [x_0, y_0, psi_0, v_0].
        U : ndarray
            Control variables u[i] for i = 0..T - 1 with rows [u_1, u_2]
        
        Returns
        =======
        ndarray
            State trajectory from control including origin
            of shape (T + 1, 4) with rows [x, y, psi, v].
        """
        U_pad = np.concatenate((U, U[-1][None]))
        _, states = control.input_output_response(
            self.io_dynamical_system, self.timestamps, U_pad.T, x_init
        )
        return states.T
    
    def get_nominal_trajectory(self, x_init, u_init):
        """Compute non-linear nominal trajectory from vehicle model
        given initial states and initial control variable.

        Parameters
        ==========
        x_init : ndarray
            Initial state [x_0, y_0, psi_0, v_0].
        u_init : ndarray
            Initial control variable [u_1, u_2].
        
        Returns
        =======
        ndarray
            Nominal state trajectory including origin
            of shape (T + 1, 4) with rows [x, y, psi, v].
        ndarray
            Nominal control trajectory of shape (T, 2).
            Has u[i] for i = 0..T - 1 with rows [u_1, u_2].
        """
        U_bar = np.repeat(u_init[None], self.T, axis=0)
        X_bar = self.states_from_control(x_init, U_bar)
        return X_bar, U_bar
    
    def get_discrete_time_ltv(self, x_init, u_init):
        """Compute discrete-time LTV approximation from vehicle model.
        
        Parameters
        ==========
        x_init : ndarray
            Initial state [x_0, y_0, psi_0, v_0].
        u_init : ndarray
            Initial control variable [u_1, u_2].
        
        Returns
        =======
        ndarray
            Nominal state trajectory including origin of shape (T + 1, 4).
            Has u[i] for i = 0..T with rows [x, y, psi, v].
        ndarray
            Nominal control trajectory of shape (T, 2).
            Has u[i] for i = 0..T - 1 with rows [u_1, u_2].
        list of ndarray
            State matrix. Has A[i] for i = 0..T - 1.
        list of ndarray
            Input matrix. Has B[i] for i = 0..T - 1.
        """
        X_bar, U_bar = self.get_nominal_trajectory(x_init, u_init)
        C = get_output_matrix()
        D = get_feedforward_matrix()
        As = []
        Bs = []
        for i in range(self.T):
            A = get_state_matrix(X_bar[i], U_bar[i], l_r=self.l_r, L=self.L)
            B = get_input_matrix(X_bar[i], U_bar[i], l_r=self.l_r, L=self.L)
            sys = control.matlab.c2d(control.matlab.ss(A, B, C, D), self.Ts)
            As.append(sys.A)
            Bs.append(sys.B)
        return X_bar, U_bar, As, Bs
    
    def get_optimization_ltv(self, x_init, u_init):
        """Compute discrete-time LTV approximation from vehicle model
        for optimization. To compute state trajectory from control trajectory
        do the following:

        ```
        u = U.ravel()
        u_delta = u - u_bar
        x = Gamma @ u_delta + x_bar
        X = x.reshape((T, nx))
        ```

        Parameters
        ==========
        x_init : ndarray
            Initial state [x_0, y_0, psi_0, v_0].
        u_init : ndarray
            Initial control variable [u_1, u_2].
        
        Returns
        =======
        ndarray
            Nominal state trajectory of shape (T, 4).
            Flattened. Has u[i] for i = 1..T with rows [x, y, psi, v].
        ndarray
            Nominal control trajectory of shape (T, 2).
            Flattened. Has u[i] for i = 0..T - 1 with rows [u_1, u_2].
        ndarray
            Matrix Gamma used to compute `x = Gamma @ u_delta + x_bar`.
        int
            Number of state variables.
        int
            Number of control variables.
        """
        X_bar, U_bar, As, Bs = self.get_discrete_time_ltv(x_init, u_init)
        # nx, nu - number of state and control variables
        nx, nu = Bs[0].shape
        T = self.T
        # B_bar has shape (T*nx, T*nu)
        B_bar = scipy.linalg.block_diag(*Bs)
        # A_bar has shape (T*nx, T*nx)
        A_bar = np.eye(T*nx)
        A_bar[4:, :(T - 1)*nx] -= scipy.linalg.block_diag(*As[1:])
        # Gamma has shape (T*nx, T*nu)
        Gamma = np.linalg.solve(A_bar, B_bar)
        x_bar = X_bar[1:].ravel()
        u_bar = U_bar.ravel()
        return x_bar, u_bar, Gamma, nx, nu
