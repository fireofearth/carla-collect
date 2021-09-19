"""This is for Trajectron data wrangling."""

import os
import json
import logging
import glob

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

import carla
import utility as util
from ..label import carla_id_maker
from ..map import MapDataExtractor
from ... import CACHEDIR


def node_to_df(node):
    columns = ["_".join(t) for t in node.data.header]
    return pd.DataFrame(node.data.data, columns=columns)


def scene_to_df(scene):
    dfs = [node_to_df(node) for node in scene.nodes if repr(node.type) == "VEHICLE"]
    tmp_dfs = []
    for node, df in zip(scene.nodes, dfs):
        df.insert(0, "node_id", str(node.id))
        df.insert(0, "frame_id", range(len(df)))
        tmp_dfs.append(df)
    return pd.concat(tmp_dfs)


def vertex_set_to_smpoly(vertex_set):
    polygons = []
    for vertices in vertex_set:
        polygons.append([vertices, []])
    return shapely.geometry.MultiPolygon(polygons)


def vertices_to_smpoly(vertices):
    polygons = [[vertices, []]]
    return shapely.geometry.MultiPolygon(polygons)


def trajectory_curvature(t):
    path_distance = np.linalg.norm(t[-1] - t[0])
    lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
    path_length = np.sum(lengths)
    if np.isclose(path_distance, 0.0):
        return 0, 0, 0
    return (path_length / path_distance) - 1, path_length, path_distance


class FrequencyModificationConfig(dict):
    """Values to set frequency modifier of scenes."""

    def __init__(
        self,
        complete_intersection=4,
        stopped_car=1,
        at_intersection=2,
        major_turn=18,
        minor_turn=7,
        other=1,
    ):
        super().__init__(
            complete_intersection=complete_intersection,
            stopped_car=stopped_car,
            at_intersection=at_intersection,
            major_turn=major_turn,
            minor_turn=minor_turn,
            other=other,
        )
        self.__dict__ = self

    @classmethod
    def from_file(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(**config)


class TrajectronDataToLabel(object):
    # Directory of cache
    MAP_NAMES = ["Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
    # radius used to check whether vehicle is in junction
    TLIGHT_DETECT_RADIUS = 25.0
    STOPPED_CAR_TOL = 1.0

    def __init__(self, config):
        self.config = config
        for map_name in self.MAP_NAMES:
            cachepath = f"{CACHEDIR}/map_data.{map_name}.pkl"
            with open(cachepath, "rb") as f:
                payload = dill.load(f, encoding="latin1")
            self.map_datum[map_name] = util.AttrDict(
                road_polygons=payload["road_polygons"],
                white_lines=payload["white_lines"],
                yellow_lines=payload["yellow_lines"],
                junctions=payload["junctions"],
            )

    def set_node_frequency_multiplier(self, env, fm_modification):
        """Mutates environment by setting the frequency modifier of scenes."""

        # map to scene+node ID to node in scene
        maps_ids_nodes_dict = {}
        # map to scene+node ID to node Shapely LineString
        ids_sls_dict = {}
        logging.info("Getting trajectories of vehicles from every scene.")
        for scene in tqdm(env.scenes):
            for node in scene.nodes:
                map_name = carla_id_maker.extract_value(scene.name, "map_name")
                scene_node_id = scene.name + "/" + node.id
                util.setget_dict_from_dict(maps_ids_nodes_dict, map_name)[
                    scene_node_id
                ] = node
                node_df = node_to_df(node)
                pos = node_df[["position_x", "position_y"]].values.copy()
                pos += np.array([scene.x_min, scene.y_min])
                sls = shapely.geometry.LineString(pos)
                ids_sls_dict[scene_node_id] = sls

        # collect shapes of all the intersections
        # map to Shapely MultiPolygon for junction entrance/exits
        map_to_smpolys = {}
        # map to Shapely Circle covering junction region
        map_to_scircles = {}
        for map_name in self.MAP_NAMES:
            for _junction in self.map_datum[map_name].junctions:
                f = lambda x, y, yaw, l: util.vertices_from_bbox(
                    np.array([x, y]), yaw, np.array([5.0, 0.95 * l])
                )
                vertex_set = util.map_to_ndarray(
                    lambda wps: util.starmap(f, wps), _junction.waypoints
                )
                smpolys = util.map_to_list(vertex_set_to_smpoly, vertex_set)
                util.setget_dict_from_dict(map_to_smpolys, map_name).extend(smpolys)
                x, y = _junction.pos
                scircle = shapely.geometry.Point(x, y).buffer(self.TLIGHT_DETECT_RADIUS)
                util.setget_dict_from_dict(map_to_scircles, map_name).append(scircle)

        logging.info("Labelling node data.")
        counts = util.AttrDict(
            at_intersection=0,
            stopped_car=0,
            complete_intersection=0,
            major_turn=0,
            minor_turn=0,
            all=0,
            other=0,
        )

        with tqdm(total=len(env.scenes)) as pbar:
            for map_name, ids_nodes_dict in maps_ids_nodes_dict.items():
                for scene_node_id, node in ids_nodes_dict.items():
                    is_at_intersection = False
                    is_stopped_car = False
                    # vehicle completely crossed intersection
                    # (trajectory enters and exits the intersection)
                    is_complete_intersection = False
                    is_major_turn = False
                    is_minor_turn = False

                    sls = ids_sls_dict[scene_node_id]
                    for smpoly in map_to_smpolys[map_name]:
                        spoly_enter = util.select(smpoly.geoms, 0)
                        spoly_exit = util.select(smpoly.geoms, 1)
                        res = sls.intersection(spoly_enter)
                        if not res.is_empty:
                            res = sls.intersection(spoly_exit)
                            if not res.is_empty:
                                is_complete_intersection = True
                                break

                    for scircle in map_to_scircles[map_name]:
                        res = sls.intersection(scircle)
                        if not res.is_empty:
                            is_at_intersection = True
                            break

                    if sls.length < self.STOPPED_CAR_TOL:
                        is_stopped_car = True
                    else:
                        curvature, _, _ = trajectory_curvature(np.array(sls.coords))
                        if curvature > 0.1:
                            is_major_turn = True
                        if curvature > 0.01:
                            is_minor_turn = True

                    counts.all += 1
                    if is_complete_intersection:
                        counts.complete_intersection += 1
                        node.frequency_multiplier = (
                            fm_modification.complete_intersection
                        )
                    elif is_stopped_car:
                        counts.stopped_car += 1
                        node.frequency_multiplier = fm_modification.stopped_car
                    elif is_at_intersection:
                        counts.at_intersection += 1
                        node.frequency_multiplier = fm_modification.at_intersection
                    elif is_major_turn:
                        counts.major_turn += 1
                        node.frequency_multiplier = fm_modification.major_turn
                    elif is_minor_turn:
                        counts.minor_turn += 1
                        node.frequency_multiplier = fm_modification.minor_turn
                    else:
                        counts.other += 1
                        node.frequency_multiplier = fm_modification.other

                    pbar.update(1)

        return counts


class TrajectronTown03DataToLabel(object):
    """Trajectron++ data extracted from Town03 ONLY to label."""

    # Directory of cache
    CACHEDIR = "cache"
    # radius used to check whether vehicle is in junction
    TLIGHT_DETECT_RADIUS = 25.0
    STOPPED_CAR_TOL = 1.0

    def __extract_data(self):
        client = carla.Client(self.config.host, self.config.port)
        client.set_timeout(10.0)
        carla_world = client.get_world()
        carla_map = carla_world.get_map()
        if carla_map.name != "Town03":
            raise Exception("Currently only able to extract map data from Town03")
        extractor = MapDataExtractor(carla_world, carla_map)
        p = extractor.extract_road_polygons_and_lines()
        road_polygons, yellow_lines, white_lines = (
            p.road_polygons,
            p.yellow_lines,
            p.white_lines,
        )
        junctions = extractor.extract_junction_with_portals()
        payload = {
            "road_polygons": road_polygons,
            "yellow_lines": yellow_lines,
            "white_lines": white_lines,
            "junctions": junctions,
        }
        os.makedirs(self.CACHEDIR, exist_ok=True)
        with open(self.cachepath, "wb") as f:
            dill.dump(payload, f, protocol=dill.HIGHEST_PROTOCOL)
        return payload

    def __init__(self, config):
        self.config = config
        self.cachepath = f"{self.CACHEDIR}/map_data.Town03.pkl"
        self.map_name = "Town03"

        try:
            with open(self.cachepath, "rb") as f:
                payload = dill.load(f, encoding="latin1")
            _, _, _, _ = (
                payload["road_polygons"],
                payload["white_lines"],
                payload["yellow_lines"],
                payload["junctions"],
            )
        except Exception as e:
            logging.warning(e)
            logging.info("Extracting map data from CARLA Simulator")
            payload = self.__extract_data()
        self.road_polygons = payload["road_polygons"]
        self.white_lines = payload["white_lines"]
        self.yellow_lines = payload["yellow_lines"]
        self.junctions = payload["junctions"]

        if config.debug:
            self.__display_map()

    def __display_map(self):
        # Plot all the junctions and entrance/exits to/from the junctions
        fig, ax = plt.subplots(figsize=(20, 20))

        for poly in self.road_polygons:
            patch = patches.Polygon(poly[:, :2], fill=True, color="grey")
            ax.add_patch(patch)
        for line in self.yellow_lines:
            ax.plot(line.T[0], line.T[1], c="yellow", linewidth=2)
        for line in self.white_lines:
            ax.plot(line.T[0], line.T[1], c="white", linewidth=2)

        for idx, junction in enumerate(self.junctions):
            x, y = junction["pos"]
            ax.plot(x, y, "ro", markersize=10)
            ax.text(x + 5, y - 5, str(idx), color="r", size=20)
            circ = patches.Circle(
                (
                    x,
                    y,
                ),
                radius=25,
                color="g",
                fc="none",
            )
            ax.add_patch(circ)
            for _wp1, _wp2 in junction["waypoints"]:
                # entrances
                x, y, yaw, lane_width = _wp1
                lw = np.array([5.0, lane_width])
                bbox = util.vertices_from_bbox(np.array([x, y]), yaw, lw)
                bb = patches.Polygon(bbox, closed=True, color="b", fc="none")
                ax.add_patch(bb)
                # exits
                x, y, yaw, lane_width = _wp2
                lw = np.array([5.0, lane_width])
                bbox = util.vertices_from_bbox(np.array([x, y]), yaw, lw)
                bb = patches.Polygon(bbox, closed=True, color="r", fc="none")
                ax.add_patch(bb)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        fig.tight_layout()
        plt.show()

    def hardcode_set_node_frequency_multiplier(self, env):
        """
        Mutates environment

        TODO: hardcoded. Figure out a way to do this in more flexible way?
        """
        # fm_modification = util.AttrDict(
        #         complete_intersection=3,
        #         stopped_car = 1,
        #         at_intersection=2,
        #         major_turn=25,
        #         minor_turn=10,
        #         other=2)
        fm_modification = util.AttrDict(
            complete_intersection=4,
            stopped_car=1,
            at_intersection=2,
            major_turn=18,
            minor_turn=7,
            other=1,
        )

        # scene+node ID to node in scene
        id_to_node = {}
        # scene+node ID to node Shapely LineString
        id_to_node_sls = {}
        logging.info("Getting trajectories of vehicles from every scene.")
        for scene in tqdm(env.scenes):
            for node in scene.nodes:
                scene_node_id = scene.name + "/" + node.id
                id_to_node[scene_node_id] = node
                node_df = node_to_df(node)
                pos = node_df[["position_x", "position_y"]].values.copy()
                pos += np.array([scene.x_min, scene.y_min])
                sls = shapely.geometry.LineString(pos)
                id_to_node_sls[scene_node_id] = sls

        # collect shapes of all the intersections
        smpolys = []
        scircles = []
        for _junction in self.junctions:
            f = lambda x, y, yaw, l: util.vertices_from_bbox(
                np.array([x, y]), yaw, np.array([5.0, 0.95 * l])
            )
            vertex_set = util.map_to_ndarray(
                lambda wps: util.starmap(f, wps), _junction.waypoints
            )
            smpolys += util.map_to_list(vertex_set_to_smpoly, vertex_set)
            x, y = _junction.pos
            scircle = shapely.geometry.Point(x, y).buffer(self.TLIGHT_DETECT_RADIUS)
            scircles.append(scircle)

        logging.info("Labelling node data.")
        counts = util.AttrDict(
            at_intersection=0,
            stopped_car=0,
            complete_intersection=0,
            major_turn=0,
            minor_turn=0,
            all=0,
            other=0,
        )
        for scene_node_id, node in tqdm(id_to_node.items()):
            is_at_intersection = False
            is_stopped_car = False
            # vehicle completely crossed intersection (trajectory enters and exits the intersection)
            is_complete_intersection = False
            is_major_turn = False
            is_minor_turn = False

            sls = id_to_node_sls[scene_node_id]
            for smpoly in smpolys:
                spoly_enter = util.select(smpoly.geoms, 0)
                spoly_exit = util.select(smpoly.geoms, 1)
                res = sls.intersection(spoly_enter)
                if not res.is_empty:
                    res = sls.intersection(spoly_exit)
                    if not res.is_empty:
                        is_complete_intersection = True
                        break

            for scircle in scircles:
                res = sls.intersection(scircle)
                if not res.is_empty:
                    is_at_intersection = True
                    break

            if sls.length < self.STOPPED_CAR_TOL:
                is_stopped_car = True
            else:
                curvature, _, _ = trajectory_curvature(np.array(sls.coords))
                if curvature > 0.1:
                    is_major_turn = True
                if curvature > 0.01:
                    is_minor_turn = True

            counts.all += 1
            if is_complete_intersection:
                counts.complete_intersection += 1
                node.frequency_multiplier = fm_modification.complete_intersection
            elif is_stopped_car:
                counts.stopped_car += 1
                node.frequency_multiplier = fm_modification.stopped_car
            elif is_at_intersection:
                counts.at_intersection += 1
                node.frequency_multiplier = fm_modification.at_intersection
            elif is_major_turn:
                counts.major_turn += 1
                node.frequency_multiplier = fm_modification.major_turn
            elif is_minor_turn:
                counts.minor_turn += 1
                node.frequency_multiplier = fm_modification.minor_turn
            else:
                counts.other += 1
                node.frequency_multiplier = fm_modification.other

            # Do independently tally
            # if is_complete_intersection:
            #     counts.complete_intersection += 1
            # if is_stopped_car:
            #     counts.stopped_car += 1
            # if is_at_intersection:
            #     counts.at_intersection += 1
            # if is_major_turn:
            #     counts.major_turn += 1
            # if is_minor_turn:
            #     counts.minor_turn += 1

        return counts

    def classify_label_to_scene_ids(self, sample_dict):
        """
        TODO: not being used

        Parameters
        ==========
        sample_dict : dict of (str: Scene)
            Mapping of scene ID and the scene itself.

        Returns
        =======
        list of str
            The filtered scene IDs.
        """
        scene_id_to_ego_sls = {}
        logging.info("Getting trajectories of ego vehicles from every scene.")
        for scene_id, scene in tqdm(sample_dict.items()):
            scene_df = scene_to_df(scene)
            node_df = scene_df[scene_df["node_id"] == "ego"]
            pos = node_df[["position_x", "position_y"]].values
            sls = shapely.geometry.LineString(pos)
            scene_id_to_ego_sls[scene_id] = sls

        # collect shapes of all the intersections
        smpolys = []
        scircles = []
        for _junction in self.junctions:
            f = lambda x, y, yaw, l: util.vertices_from_bbox(
                np.array([x, y]), yaw, np.array([5.0, 0.95 * l])
            )
            vertex_set = util.map_to_ndarray(
                lambda wps: util.starmap(f, wps), _junction["waypoints"]
            )
            smpolys += util.map_to_list(vertex_set_to_smpoly, vertex_set)
            x, y = _junction["pos"]
            scircle = shapely.geometry.Point(x, y).buffer(self.TLIGHT_DETECT_RADIUS)
            scircles.append(scircle)

        label_to_scene_ids = {"is_at_intersecion": [], "other": []}
        for scene_id, sls in tqdm(scene_id_to_ego_sls.items()):
            is_at_intersection = False
            is_stopped_car = False

            for scircle in scircles:
                res = sls.intersection(scircle)
                if not res.is_empty:
                    is_at_intersection = True
                    break

            if sls.length < self.STOPPED_CAR_TOL:
                is_stopped_car = True

            if is_at_intersection:
                label_to_scene_ids["is_at_intersecion"].append(scene_id)
            else:
                label_to_scene_ids["other"].append(scene_id)

        return label_to_scene_ids
