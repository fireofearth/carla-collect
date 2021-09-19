"""Cache the map from CARLA Simulator."""

import logging
import argparse
import os
import dill

logging.basicConfig(
    format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO
)

import carla
from collect import CACHEDIR
from collect.generate.map import CARLA_MAP_NAMES, MapDataExtractor


def parse_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    return argparser.parse_args()


def main():
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    carla_world = client.get_world()
    carla_map = carla_world.get_map()
    for map_name in CARLA_MAP_NAMES:
        logging.info(f"Caching map data from {map_name}.")
        if carla_map.name != map_name:
            carla_world = client.load_world(map_name)
            carla_map = carla_world.get_map()
        extractor = MapDataExtractor(carla_world, carla_map)
        logging.info("    Extracting and caching road polygons and dividers")
        p = extractor.extract_road_polygons_and_lines()
        road_polygons, yellow_lines, white_lines = (
            p.road_polygons,
            p.yellow_lines,
            p.white_lines,
        )
        logging.info("    Extracting and caching road junctions")
        junctions = extractor.extract_junction_with_portals()
        logging.info("    Extracting and caching spawn points")
        spawn_points = extractor.extract_spawn_points()
        payload = {
            "road_polygons": road_polygons,
            "yellow_lines": yellow_lines,
            "white_lines": white_lines,
            "junctions": junctions,
            "spawn_points": spawn_points,
        }
        os.makedirs(CACHEDIR, exist_ok=True)
        cachepath = f"{CACHEDIR}/map_data.{map_name}.pkl"
        with open(cachepath, "wb") as f:
            dill.dump(payload, f, protocol=dill.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
