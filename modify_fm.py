"""Modify node/scene frequency multiplier.

I only need to connect to the CARLA server once to extract map data if my cache is empty.
"""
import pprint
import os
import logging
import argparse
import shutil
import random
from glob import glob
import re

import dill
import numpy as np
import pandas as pd
from tqdm import tqdm

import utility as util
import utility.arguments as uarguments

from collect import OUTDIR
from collect.generate.scene.v3.trajectron_scene import augment_scene
from collect.generate.dataset import SampleGroupCreator, CrossValidationSplitCreator
from collect.generate.scene.trajectron_util import make_environment
from collect.generate.dataset.trajectron import (
    TrajectronSceneData,
    # TrajectronDataToLabel,
    FrequencyModificationConfig,
)

pp = pprint.PrettyPrinter(indent=4)
logger = logging.getLogger(__file__)


def parse_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="Show debug information",
    )
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
    argparser.add_argument(
        "--data-path",
        default=".",
        type=uarguments.file_path,
        help=".pkl data file containing scenes.",
    )
    argparser.add_argument(
        "--dry-run", action="store_true", help="Don't modify the frequency modifier"
    )
    argparser.add_argument(
        "--modifier",
        default="modifier.json",
        type=uarguments.file_path,
        help="Path of the frequency modifier",
    )
    return argparser.parse_args()


def main(config):
    logging.info(f"Loading dataset {config.data_path}")
    with open(config.data_path, "rb") as f:
        env = dill.load(f, encoding="latin1")

    logger.info("Inspecting scene data.")
    scene_data = TrajectronSceneData(env.scenes)
    scene_data.log_node_count(filename="dataset_scene_histogram")

    if config.dry_run:
        logging.info("Done dry run.")
    else:
        fm_modification = FrequencyModificationConfig.from_file(config.modifier)
        logging.info(f"Using config to set frequency mod: {fm_modification}.")
        scene_data.set_node_fm(fm_modification)
        savepath = re.sub(r"\.pkl$", "_modfm.pkl", config.data_path)
        logging.info(f"Saving modified dataset {savepath}")
        with open(savepath, "wb") as f:
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
        logging.info("Done.")


if __name__ == "__main__":
    config = parse_arguments()
    log_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s",
        level=log_level
    )
    main(config)
