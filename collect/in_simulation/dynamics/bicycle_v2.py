import math

import numpy as np
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

def get_state_matrix(x, y, psi, v, delta, l_r=0.5, L=1.0):
    beta   = get_beta(delta, l_r=l_r, L=L)
    df3_dv = (1/L)*cos(beta)*tan(delta)
    return np.array([
        # x, y, psi, v
        [0, 0, -v*sin(psi + beta), cos(psi + beta)],
        [0, 0,  v*cos(psi + beta), sin(psi + beta)],
        [0, 0,                  0,          df3_dv],
        [0, 0,                  0,               0],
    ])

def get_input_matrix(x, y, psi, v, delta, l_r=0.5, L=1.0):
    beta   = get_beta(delta, l_r=l_r, L=L)
    dbeta  = get_dbeta_ddelta(delta, l_r=l_r, L=L)
    tan2   = tan(delta)**2
    return np.array([
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
