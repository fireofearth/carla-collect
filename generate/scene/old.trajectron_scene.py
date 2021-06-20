
"""
Dummy file to make unpickle work properly
"""

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

curv_0_2 = 0
curv_0_1 = 0
total = 0
occlusion = 0

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

def print_and_reset_specs():
    pass

def augment_scene(scene, angle):
    pass

def augment(scene):
    pass

def trajectory_curvature(t):
    pass

def plot_occlusion(scene, data, node_df, occl_count):
    pass

def process_trajectron_scene(scene, data, max_timesteps, scene_config):
    pass


def plot_trajectron_scene(savedir, scene):
    pass


class TrajectronPlusPlusSceneBuilder(SceneBuilder):
    DISTANCE_FROM_ROAD = 20.

    def __process_carla_scene(self, data):
        pass

    def process_scene(self, data):
        pass
