import numpy as np
import pandas as pd

def node_to_df(node):
    """Trajectron++ node to data frame."""
    columns = ["_".join(t) for t in node.data.header]
    return pd.DataFrame(node.data.data, columns=columns)


def scene_to_df(scene):
    """Trajectron++ scene to data frame."""
    dfs = [node_to_df(node) for node in scene.nodes if repr(node.type) == "VEHICLE"]
    tmp_dfs = []
    for node, df in zip(scene.nodes, dfs):
        df.insert(0, "node_id", str(node.id))
        df.insert(0, "frame_id", range(len(df)))
        tmp_dfs.append(df)
    return pd.concat(tmp_dfs)


def scenes_to_df(scenes, use_world_position=False):
    """Combine Trajectron++ scenes to a single data frame."""
    dfs = []
    for scene in scenes:
        df = scene_to_df(scene)
        df["scene_id"] = scene.name
        df['node_id'] = scene.name + '/' + df['node_id']
        if use_world_position:
            df[['position_x', 'position_y']] += np.array([scene.x_min, scene.y_min])
        dfs.append(df)
    return pd.concat(dfs)