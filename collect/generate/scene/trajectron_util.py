import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    # trajectron-plus-plus/trajectron
    from environment import Environment
except ModuleNotFoundError as e:
    raise Exception("You forgot to link trajectron-plus-plus/trajectron")

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
    env = Environment(node_type_list=['VEHICLE'],
                name=name,
                standardization=standardization)
    attention_radius = dict()
    attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0
    env.attention_radius = attention_radius
    env.robot_type = env.NodeType.VEHICLE
    return env
