"""Module for generating and inspecting Trajectron++ specific datasets."""

# Built-in libraries
import os
import json
import logging

# PyPI libraries
import dill
from tqdm import tqdm
import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patheffects as pe

# Local libararies
import carla
import utility as util

# Modules
from ..label import carla_id_maker
from ..map import CachedMapData
from ... import CACHEDIR, OUTDIR

logger = logging.getLogger(__name__)


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


def vertex_set_to_smpoly(vertex_set):
    polygons = []
    for vertices in vertex_set:
        polygons.append([vertices, []])
    return shapely.geometry.MultiPolygon(polygons)


def vertices_to_smpoly(vertices):
    polygons = [[vertices, []]]
    return shapely.geometry.MultiPolygon(polygons)


def trajectory_curvature(t):
    """Compute trajectory curvature based on Trajectron++ code."""
    path_distance = np.linalg.norm(t[-1] - t[0])
    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.0):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


def max_curvature(X):
    """Compute trajectory curvature based on cubic spline approx
    and applying convolution to impove numerical stability."""
    spline, _, distances = util.npu.interp_spline(X, normalize_interp=False, tol=0.1)
    length = distances[-1]
    s = np.linspace(0, length, 100)
    ddspline = spline.derivative(2)
    k = np.linalg.norm(ddspline(s), axis=1)
    k_blur, _ = util.npu.apply_kernel_1d(k, util.npu.kernel_1d_gaussian, 5, 3)
    k_max = np.max(k_blur)
    return k_max


class FrequencyModificationConfig(dict):
    """Values to set frequency modifier of scenes."""

    def __init__(
        self,
        complete_intersection : int=1,
        significant_at_intersection : int=1,
        stopped_at_intersection : int=1,
        other_at_intersection : int=1,
        turn_at_other : int=1,
        significant_at_other : int=1,
        stopped_at_other : int=1,
        other_at_other : int=1,
    ):
        super().__init__(
            complete_intersection=complete_intersection,
            significant_at_intersection=significant_at_intersection,
            stopped_at_intersection=stopped_at_intersection,
            other_at_intersection=other_at_intersection,
            turn_at_other=turn_at_other,
            significant_at_other=significant_at_other,
            stopped_at_other=stopped_at_other,
            other_at_other=other_at_other,
        )
        self.__dict__ = self

    @classmethod
    def from_file(cls, config_path):
        """Read a frequency modifier from JSON file."""
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(**config)


# Maps sets of node attributes to sets of mutually exclusive node/scene attributes
NODEATTR_SCENEATTR_MAP = [
    ### counts in intersection
    ( ## mapping
        # node attribute
        ["is_complete_intersection"],
        # scene attribute
        ["complete_intersection"]
    ),(
        ["is_at_intersection", "is_significant_car"],
        ["significant_at_intersection"]
    ),(
        ["is_at_intersection", "is_stopped_car"],
        ["stopped_at_intersection"]
    ),(
        ["is_at_intersection"],
        ["other_at_intersection"]
    ),
    ### counts outside of intersection (other)
    (
        ["is_major_turn", "is_significant_car"],
        ["turn_at_other"]
    ),(
        ["is_minor_turn", "is_significant_car"],
        ["turn_at_other"]
    ),(
        ["is_significant_car"],
        ["significant_at_other"]
    ),(
        ["is_stopped_car"],
        ["stopped_at_other"]
    ),(
        [],
        ["other_at_other"]
    )
]

# List of scene attributes
SCENEATTRS = util.deduplicate(
    util.merge_list_of_list(util.map_to_list(util.second, NODEATTR_SCENEATTR_MAP))
)

class TrajectronSceneData(object):
    """Trajectron++ scene data. Used to inspect data, count samples in the dataset,
    and reweight samples when generating a dataset.
    
    Attributes
    ==========
    n_nodes : int
        Number of nodes.
    nodeid_scene_dict : dict of (str, Scene)
        scene+node ID => scene the node is in
    sceneid_scene_dict : dict of (str, Scene)
        scene ID => scene
    sceneid_count_dict : dict of (str, util.AttrDict)
        scene ID => scene counts
    map_nodeids_dict : dict of (str, list of str)
        map to scene+node ID => node
    nodeid_node_dict : dict of (str, Node)
        node ID => node
    nodeid_sls_dict : dict of (str, shapely.geometry.LineString)
        scene+node ID to node Shapely LineString
    total_counts : util.AttrDict
        total counts of nodes across the dataset.
    nodeattr_df : DataFrame
        data frame of node attributes.
    sceneattr_count_df : DataFrame
        Each row is the count of nodes in the scene.
    scene_count_df : DataFrame
        Breakdown of nodes per scene.
    scenes : list of Scene
        List of scenes in dataset.
    cached_map_data : CachedMapData
        Cached map data for querying.
    """
    STOPPED_CAR_TOL = 1.0
    COUNTS_TEMPLATE = util.AttrDict(all=0, **{attr: 0 for attr in SCENEATTRS})

    def __construct_mappings_from_node(self):
        """Contstruct mappings from node."""
        self.n_nodes = 0
        self.nodeid_scene_dict = {}
        self.sceneid_scene_dict = {}
        self.sceneid_count_dict = {}
        self.map_nodeids_dict = {}
        self.nodeid_node_dict = {}
        self.nodeid_sls_dict = {}
        logger.info("Getting trajectories of vehicles from every scene.")
        for scene in tqdm(self.scenes):
            pos_adjust = np.array([scene.x_min, scene.y_min])
            map_name = carla_id_maker.extract_value(scene.name, "map")
            self.sceneid_count_dict[scene.name] = self.COUNTS_TEMPLATE.copy()
            self.sceneid_count_dict[scene.name].scene_id = scene.name
            self.sceneid_scene_dict[scene.name] = scene
            for node in scene.nodes:
                nodeid = scene.name + "/" + node.id
                self.nodeid_scene_dict[nodeid] = scene
                self.nodeid_node_dict[nodeid] = node
                util.setget_list_from_dict(
                    self.map_nodeids_dict, map_name
                ).append(nodeid)
                node_df = node_to_df(node)
                pos = node_df[["position_x", "position_y"]].values + pos_adjust
                sls = shapely.geometry.LineString(pos)
                self.nodeid_sls_dict[nodeid] = sls
                self.n_nodes += 1
    
    def __inspect_intersection_completedness(self, map_name, nodeid):
        """Whether a car completed an intersection
        (trajectory enters and exits the intersection)
        by checking containment twice."""
        sls = self.nodeid_sls_dict[nodeid]
        for smpoly in self.cached_map_data.map_to_smpolys[map_name]:
            spoly_enter = util.select(smpoly.geoms, 0)
            spoly_exit  = util.select(smpoly.geoms, 1)
            res = sls.intersection(spoly_enter)
            if not res.is_empty:
                res = sls.intersection(spoly_exit)
                if not res.is_empty:
                    return True
        return False
    
    def __inspect_intersectedness(self, map_name, nodeid):
        """Whether a car is in the intersection,
        and what its behavior is in it
        by checking intersection."""
        sls = self.nodeid_sls_dict[nodeid]
        for scircle in self.cached_map_data.map_to_scircles[map_name]:
            res = sls.intersection(scircle)
            if not res.is_empty:
                return True
        return False
    
    def __inspect_distance(self, map_name, nodeid):
        """Whether a car is moving significantly
        (at least 10m and 10 steps) by checking line properties."""
        is_significant_car = False
        is_stopped_car = False
        sls = self.nodeid_sls_dict[nodeid]
        S = len(sls.coords)
        L = sls.length
        if S >= 10 and L >= 10.0:
            is_significant_car = True
        elif L < self.STOPPED_CAR_TOL:
            is_stopped_car = True
        return (
            is_significant_car,
            is_stopped_car
        )
    
    def __inspect_curvature(self, map_name, nodeid):
        """Inspect curvature of trajectory.
        A major turn has curvature over 0.1.
        A minor turn has curvature between 0.01, 0.1.
        """
        is_major_turn = False
        is_minor_turn = False
        sls = self.nodeid_sls_dict[nodeid]
        X = np.array(sls.coords)
        try:
            k_max = max_curvature(X)
            if k_max >= 0.1:
                is_major_turn = True
            elif k_max >= 0.01:
                is_minor_turn = True
        except ValueError:
            pass
        return (
            is_major_turn,
            is_minor_turn
        )
    
    def __inspect_node(self, map_name, nodeid):
        """Label the given node by inspecting the node's properties."""
        (
            is_complete_intersection
        ) = self.__inspect_intersection_completedness(map_name, nodeid)
        (
            is_at_intersection
        ) = self.__inspect_intersectedness(map_name, nodeid)
        (
            is_significant_car,
            is_stopped_car
        ) = self.__inspect_distance(map_name, nodeid)
        (
            is_major_turn,
            is_minor_turn
        ) = self.__inspect_curvature(map_name, nodeid)
        return util.AttrDict(
            node_id=nodeid,
            is_complete_intersection=is_complete_intersection,
            is_at_intersection=is_at_intersection,
            is_significant_car=is_significant_car,
            is_stopped_car=is_stopped_car,
            is_major_turn=is_major_turn,
            is_minor_turn=is_minor_turn
        )
    
    def __inspect_nodes(self):
        """Inspect each node and gather node attributes."""
        logger.info("Inspecting node data.")
        nodeattrs = []
        with tqdm(total=self.n_nodes) as pbar:
            for map_name, nodeids in self.map_nodeids_dict.items():
                for nodeid in nodeids:
                    nodeattr = self.__inspect_node(map_name, nodeid)
                    nodeattrs.append(nodeattr)
                    pbar.update(1)
        return nodeattrs
    
    def __count_node(self, nodeattr):
        """Update count of a scene given a node."""
        nodeid = nodeattr.node_id
        counts = self.sceneid_count_dict[self.nodeid_scene_dict[nodeid].name]
        for _counts in [counts, self.total_counts]:
            _counts.all += 1
            for _nodeattrnames, _sceneattrnames in NODEATTR_SCENEATTR_MAP:
                if all([nodeattr[_nodeattrname] for _nodeattrname in _nodeattrnames]):
                    for _sceneattrname in _sceneattrnames:
                        _counts[_sceneattrname] += 1
                    break
    
    def __count_nodes(self, nodeattrs):
        """Count nodes in each scene."""
        logger.info("Counting node data.")
        self.total_counts = self.COUNTS_TEMPLATE.copy()
        self.nodeattr_df = pd.DataFrame.from_records(nodeattrs)
        for i, nodeattr in self.nodeattr_df.iterrows():
            self.__count_node(nodeattr)
        # sceneattr_count_df : DataFrame
        #   Each row is the count of nodes in the scene.
        self.scene_count_df = pd.DataFrame.from_records(iter(self.sceneid_count_dict.values()))
    
    def __extract_data(self):
        """Extracting scene data."""
        logger.info("Extracting scene data.")
        self.__construct_mappings_from_node()
        nodeattrs = self.__inspect_nodes()
        self.__count_nodes(nodeattrs)
                
    def __init__(self, scenes):
        self.scenes = scenes
        self.cached_map_data = CachedMapData()
        self.__extract_data()

    def inspect_node(self, scene, node):
        """Inspect attribute from node.

        Parameters
        ==========
        node : Node
            Node to inspect.
        scene : Scene
            Scene the node belongs to.

        Returns
        =======
        util.AttrDict
            The node attributes.
        """
        map_name = carla_id_maker.extract_value(scene.name, "map")
        nodeid = scene.name + "/" + node.id
        nodeattr = self.__inspect_node(map_name, nodeid)
        
        for _nodeattrnames, _sceneattrnames in NODEATTR_SCENEATTR_MAP:
            if all([nodeattr[_nodeattrname] for _nodeattrname in _nodeattrnames]):
                nodeattr.classification = _sceneattrnames.copy()
                break
        return nodeattr

    def set_node_fm(self, fm_modification: FrequencyModificationConfig):
        """Set node frequency multiplier."""
        for i, nodeattr in self.nodeattr_df.iterrows():
            nodeid = nodeattr.node_id
            node = self.nodeid_node_dict[nodeid]
            
            for _nodeattrnames, _sceneattrnames in NODEATTR_SCENEATTR_MAP:
                if all([nodeattr[_nodeattrname] for _nodeattrname in _nodeattrnames]):
                    for _sceneattrname in _sceneattrnames:
                        node.frequency_multiplier = (
                            fm_modification[_sceneattrname]
                        )
                    break

    def sample_scenes(self, n_samples, scene_ids=None):
        """Weighted sampling of scene IDs without replacement.
        If sampling more than the number of IDs available, then just return all IDs.
        
        Currently samples scenes with significan number of nodes classified as:
        
        - complete_intersection
        - significant_at_intersection
        - turn_at_other
        - significant_at_other

        Parameters
        ==========
        n_samples : int
            Number of scenes to sample from.
        scene_ids : list of str
            Selected scene IDs to sample from. If not provided then just sample
            from all scene IDs provided at initialization.

        Returns
        =======
        list of str
            The sampled scenes IDs.
        """
        if scene_ids is None:
            scene_count_df = self.scene_count_df
        else:
            sceneid_count_dict = util.subdict(self.sceneid_count_dict, scene_ids)
            scene_count_df = pd.DataFrame.from_records(iter(sceneid_count_dict.values()))
        if n_samples >= len(scene_count_df):
            return list(scene_count_df["scene_id"])

        mask = np.logical_or.reduce([
            scene_count_df["complete_intersection"] > 4,
            scene_count_df["significant_at_intersection"] > 4,
            scene_count_df["turn_at_other"] > 4,
            scene_count_df["significant_at_other"] > 4,
        ])
        n = len(scene_count_df)
        n1 = np.sum(mask)
        n2 = n - n1
        a1 = 9 * n2 / n1
        logger.info(
            f"total scenes n={n}, prioritized scenes n1={n1}, deprioritized scenes n2={n2}."
        )
        logger.info(
            f"multiplier a1={a1}, expected proportion of selected {a1*n1 / (a1*n1 + n2)}."
        )
        scene_count_df.loc[mask, "weight"] = a1
        scene_count_df.loc[~mask, "weight"] = 1
        sample_df = scene_count_df.sample(n_samples, replace=False, weights="weight")
        return list(sample_df["scene_id"])
                    
    def log_node_count(self, filename="scene_histogram"):
        """Log total node counts, and save a histogram plot of node counts in scenes."""
        logger.info("node mutually-exclusive labels:")
        labels = ", ".join(self.scene_count_df.columns)
        logger.info(labels)
        logger.info("")
        logger.info("total node count:")
        n_all = self.total_counts.all
        for x in self.total_counts.items():
            logger.info(f"{x[0]}: {x[1]}, frac. {round(x[1] / n_all, 2)}")

        hist_attrs = util.AttrDict(bins=30, range=(0, 30))
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(421)
        selector = "complete_intersection"
        self.scene_count_df[selector].hist(ax=ax, **hist_attrs)
        ax.set_title(selector)

        ax = fig.add_subplot(422)
        selector = "significant_at_intersection"
        self.scene_count_df[selector].hist(ax=ax, **hist_attrs)
        ax.set_title(selector)

        ax = fig.add_subplot(423)
        selector = "stopped_at_intersection"
        self.scene_count_df[selector].hist(ax=ax, **hist_attrs)
        ax.set_title(selector)

        ax = fig.add_subplot(424)
        selector = "other_at_intersection"
        self.scene_count_df[selector].hist(ax=ax, **hist_attrs)
        ax.set_title(selector)

        ax = fig.add_subplot(425)
        selector = "turn_at_other"
        self.scene_count_df[selector].hist(ax=ax, **hist_attrs)
        ax.set_title(selector)

        ax = fig.add_subplot(426)
        selector = "significant_at_other"
        self.scene_count_df[selector].hist(ax=ax, **hist_attrs)
        ax.set_title(selector)

        ax = fig.add_subplot(427)
        selector = "stopped_at_other"
        self.scene_count_df[selector].hist(ax=ax, **hist_attrs)
        ax.set_title(selector)

        ax = fig.add_subplot(428)
        selector = "other_at_other"
        self.scene_count_df[selector].hist(ax=ax, **hist_attrs)
        ax.set_title(selector)

        for ax in fig.get_axes():
            ax.set_xlabel("Count of nodes per scene")
            ax.set_ylabel("Number of scenes")
        fig.suptitle("Histogram of node occurance in scenes")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"{filename}.png"))
        plt.close(fig)

