import math

import numpy as np

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
    l_r = params.get('l_r', 0.5)
    L = params.get('L', 1)
    psi   = x[2]
    v     = x[3]
    delta = x[4]
    u_1, u_2 = u[0], u[1]
    beta  = get_beta(delta, l_r=l_r, L=L)
    return np.array([
        v*cos(psi + beta), # dot x 
        v*sin(psi + beta), # dot y
        (v / L)*cos(beta)*tan(delta), # dot psi
        u_1, # dot v
        u_2, # dot delta
    ])

def get_state_matrix(x, y, psi, v, delta, l_r=0.5, L=1.0):
    beta   = get_beta(delta, l_r=l_r, L=L)
    dbeta  = get_dbeta_ddelta(delta, l_r=l_r, L=L)
    df3_dv = (1/L)*cos(beta)*tan(delta)
    df3_ddelta = (v/L)*(cos(beta)*(1 + tan(delta)**2) - sin(beta)*tan(delta)*dbeta)
    return np.array([
        # x, y, psi, v, delta
        [0, 0, -v*sin(psi + beta), cos(psi + beta), -v*sin(psi + beta)*dbeta],
        [0, 0,  v*cos(psi + beta), sin(psi + beta),  v*cos(psi + beta)*dbeta],
        [0, 0,                  0,          df3_dv,               df3_ddelta],
        [0, 0,                  0,               0,                        0],
        [0, 0,                  0,               0,                        0],
    ])

def get_input_matrix():
    return np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]).T

def get_output_matrix():
    return np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ])

def get_feedforward_matrix():
    return np.array([
        [0, 0],
        [0, 0],
    ])
