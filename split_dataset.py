"""Create cross-validation splits from data.

If you call synthesize.py to produce 6 pkl data files, then:

To produce full datasets do

python split_dataset.py -v --data-files $(ls out/*.pkl) --n-splits 3 --n-val 60 --n-test 60 --label v1-0

To produce small datasets do

python split_dataset.py --data-files $(ls out/*.pkl) --n-splits 3 --n-train 300 --n-val 60 --n-test 60 --label v1-0

Will produce

v1-0_split1_train.pkl
v1-0_split1_val.pkl
v1-0_split1_test.pkl
"""
import pprint
import os
import logging
import argparse
import shutil
import random
from glob import glob

import dill
import numpy as np
from tqdm import tqdm

import utility as util
import utility.arguments as uarguments

from collect import OUTDIR
from collect.generate.scene.v3.trajectron_scene import augment_scene
from collect.generate.dataset import (
    SampleGroupCreator,
    CrossValidationSplitCreator,
)
from collect.generate.dataset.trajectron import TrajectronSceneData
from collect.generate.scene.trajectron_util import make_environment

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
        "--n-groups", default=6, type=int, help="Number of groups to generate."
    )
    argparser.add_argument(
        "--n-splits",
        default=1,
        type=int,
        help="Number of split (train, val, test) sets to generate.",
    )
    argparser.add_argument(
        "--n-train",
        default=None,
        type=int,
        help="Number of training samples to include in splits. "
        "By default uses all samples.",
    )
    argparser.add_argument(
        "--n-val",
        default=None,
        type=int,
        help="Number of validation samples to include in splits. "
        "By default uses all samples.",
    )
    argparser.add_argument(
        "--n-test",
        default=None,
        type=int,
        help="Number of test samples to include in splits. "
        "By default uses all samples.",
    )
    argparser.add_argument(
        "--weighted-resampling",
        action="store_true",
        help=(
            "When selecting less than the total available samples to include in splits, "
            "do weighted random sampling that prioritizes samples with 'interesting' nodes "
            "See TrajectronSceneData.sample_scenes() for mor info."
        ),
    )
    argparser.add_argument(
        "--data-dir",
        default=".",
        type=uarguments.dir_path,
        help="Directory containing data to make splits from.",
    )
    argparser.add_argument(
        "--data-files",
        nargs="+",
        type=str,
        help="The data file paths relative to the path provided by `--data-dir`.",
    )
    argparser.add_argument(
        "--split-dir",
        default=OUTDIR,
        type=uarguments.dir_path,
        help="Directory containing data to save the splits to.",
    )
    argparser.add_argument("--label", default="carla", type=str, help="Label to tag")
    argparser.add_argument(
        "--filter",
        nargs="+",
        type=uarguments.str_kv,
        action=uarguments.ParseKVToMergeDictAction,
        default={},
        dest="filter_inclusive_labels",
        help="Words in sample ID (map, episode, agent, frame) " "to filter samples by.",
    )
    return argparser.parse_args()


"""
weighted sampling of scenes with cars that turn.
"""


def main(config):
    pp.pprint(vars(config))
    group_creator = SampleGroupCreator(config)
    split_creator = CrossValidationSplitCreator(config)

    logger.info("Loading the data.")
    envs = []
    for file_path in config.data_files:
        try:
            loadpath = os.path.join(config.data_dir, file_path)
            with open(loadpath, "rb") as f:
                env = dill.load(f, encoding="latin1")
                envs.append(env)
        except:
            logger.warning(f"Failed to load {file_path} from data directory")
            logger.warning(config.data_dir)

    env = None
    if len(envs) == 0:
        logger.warning("No data. Doing nothing.")
        return
    elif len(envs) > 0:
        logger.info(f"Loaded {len(envs)} data payloads.")
        env = envs[0]
    for _env in envs[1:]:
        env.scenes.extend(_env.scenes)
    logger.info(f"Collected {len(env.scenes)} samples.")

    sample_dict = {scene.name: scene for scene in env.scenes}
    sample_ids = list(sample_dict.keys())
    if config.weighted_resampling:
        logger.info("Inspecting scene data to use for weighted resampling.")
        scene_data = TrajectronSceneData(env.scenes)
        scene_data.log_node_count(filename="raw_scene_histogram")
    else:
        scene_data = None
    logger.info("Creating group indices of samples.")
    groups = group_creator.make_groups(sample_ids)
    logger.info("Creating split indices of samples.")
    splits = split_creator.make_splits(groups)

    augmentations_dict = dict()
    for idx, split in enumerate(splits[: config.n_splits]):
        split_idx = idx + 1
        logger.info(f"Making (train, val, test) split {split_idx}.")
        train_env = make_environment(env.name)
        val_env = make_environment(env.name)
        test_env = make_environment(env.name)

        if config.weighted_resampling:
            train_split = scene_data.sample_scenes(config.n_train, scene_ids=split[0])
        else:
            train_split = split[0][slice(config.n_train)]

        # create augmented scenes as needed
        unaugmented_ids = set(train_split) - set(augmentations_dict.keys())
        logger.info(f"Augmenting {len(unaugmented_ids)} scenes for split {split_idx}.")
        for sample_id in tqdm(unaugmented_ids):
            scene = sample_dict[sample_id]
            augmentations_dict[sample_id] = list()
            angles = np.arange(0, 360, 15)
            for angle in angles:
                augmentations_dict[sample_id].append(augment_scene(scene, angle))

        # add all existing augmentations to all scenes right before saving train set.
        for sample_id, augments in augmentations_dict.items():
            sample_dict[sample_id].augmented = augments

        train_env.scenes = util.map_to_list(sample_dict.get, train_split)
        n_train_samples = len(train_env.scenes)

        logger.info(f"Train set has {n_train_samples} scenes.")
        logger.info("Saving train set.")
        savepath = os.path.join(
            config.split_dir,
            "{label}_split{split_idx}_train.pkl".format(
                split_idx=split_idx, label=config.label
            ),
        )
        with open(savepath, "wb") as f:
            dill.dump(train_env, f, protocol=dill.HIGHEST_PROTOCOL)

        # remove all existing augmentations from all scenes right after saving train set.
        for sample_id in augmentations_dict.keys():
            del sample_dict[sample_id].augmented

        if config.weighted_resampling:
            val_split = scene_data.sample_scenes(config.n_val, scene_ids=split[1])
            test_split = scene_data.sample_scenes(config.n_test, scene_ids=split[2])
        else:
            val_split = split[1][slice(config.n_val)]
            test_split = split[2][slice(config.n_test)]
        val_env.scenes = util.map_to_list(sample_dict.get, val_split)
        n_val_samples = len(val_env.scenes)
        test_env.scenes = util.map_to_list(sample_dict.get, test_split)
        n_test_samples = len(test_env.scenes)

        # naive augmenting for each split
        # for scene in tqdm(train_env.scenes):
        #     scene.augmented = list()
        #     angles = np.arange(0, 360, 15)
        #     for angle in angles:
        #         scene.augmented.append(augment_scene(scene, angle))

        logger.info(f"Val set has {n_val_samples} scenes.")
        logger.info("Saving val set.")
        savepath = os.path.join(
            config.split_dir,
            "{label}_split{split_idx}_val.pkl".format(
                split_idx=split_idx, label=config.label
            ),
        )
        with open(savepath, "wb") as f:
            dill.dump(val_env, f, protocol=dill.HIGHEST_PROTOCOL)

        logger.info(f"Test set has {n_test_samples} scenes.")
        logger.info("Saving test set.")
        savepath = os.path.join(
            config.split_dir,
            "{label}_split{split_idx}_test.pkl".format(
                split_idx=split_idx, label=config.label
            ),
        )
        with open(savepath, "wb") as f:
            dill.dump(test_env, f, protocol=dill.HIGHEST_PROTOCOL)

    logger.info("Done.")


if __name__ == "__main__":
    config = parse_arguments()
    log_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s", level=log_level
    )
    main(config)
