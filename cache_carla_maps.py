"""Cache the map from CARLA Simulator."""

import logging
import argparse

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


def main(config):
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    MapDataExtractor.save_map_data_to_cache(client)


if __name__ == "__main__":
    config = parse_arguments()
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s", level=log_level
    )
    main(config)
