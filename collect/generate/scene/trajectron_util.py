import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    # trajectron-plus-plus/trajectron
    from environment import Environment, Scene, Node, GeometricMap, derivative_of
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

standardization = {
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}

def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])

    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        if node.type == 'PEDESTRIAN':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}

            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep)
        elif node.type == 'VEHICLE':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)
            ax = derivative_of(vx, scene.dt)
            ay = derivative_of(vy, scene.dt)

            v = np.stack((vx, vy), axis=-1)
            v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
            heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
            heading_x = heading_v[:, 0]
            heading_y = heading_v[:, 1]

            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): derivative_of(heading, dt, radian=True)}

            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=node.first_timestep,
                        non_aug_node=node)

        scene_aug.nodes.append(node)
    return scene_aug

def plot_trajectron_scene(savedir, scene):
    fig, ax = plt.subplots(figsize=(15,15))
    # extent = (scene.x_min, scene.x_max, scene.y_min, scene.y_max)
    ax.imshow(scene.map['VEHICLE'].as_image(), origin='lower')
    spectral = cm.nipy_spectral(np.linspace(0, 1, len(scene.nodes)))
    for idx, node in enumerate(scene.nodes):
        # using values from scene.map['VEHICLE'].homography
        # to scale points
        xy = 3 * node.data.data[:, :2]
        ax.scatter(xy[:, 0], xy[:, 1], color=spectral[idx])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    fig.tight_layout()
    fn = f"{ scene.name.replace('/', '_') }.png"
    fp = os.path.join(savedir, fn)
    fig.savefig(fp)

def make_environment(name):
    """Create an environment.

    Parameters
    ==========
    name : str
        Name of environment.

    Returns
    =======
    trajectron.environment.Environment
    """
    env = Environment(node_type_list=['VEHICLE'],
                name=name,
                standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
    env.attention_radius = attention_radius
    env.robot_type = env.NodeType.VEHICLE
    return env
