import os
import logging
import argparse
import shutil
import random
from glob import glob

import dill
import numpy as np
from tqdm import tqdm
from collect.generate.scene.v3.trajectron_scene import augment_scene

import utility as util
import utility.arguments as uarguments

from collect.generate.dataset import SampleGroupCreator, CrossValidationSplitCreator
from collect.generate.scene.trajectron_util import make_environment

DEFAULT_DIR = 'out'

def parse_arguments():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--n-groups',
        default=12,
        type=int,
        help="Number of groups to generate.")
    argparser.add_argument(
        '--n-splits',
        default=1,
        type=int,
        help="Number of split (train, val, test) sets to generate.")
    argparser.add_argument(
        '--data-dir',
        default=DEFAULT_DIR,
        type=uarguments.dir_path,
        help="Directory containing data to make splits from.")
    argparser.add_argument(
        '--data-files',
        nargs='+',
        type=str,
        help="The data file paths relative to the path provided by `--data-dir`.")
    argparser.add_argument(
        '--split-dir',
        default=DEFAULT_DIR,
        type=uarguments.dir_path,
        help="Directory containing data to save the splits to.")
    argparser.add_argument(
        '--label',
        default='carla',
        type=str,
        help="Label to tag"
    )
    argparser.add_argument(
        '--filter',
        nargs='+',
        type=uarguments.str_kv,
        action=uarguments.ParseKVToMergeDictAction,
        default={},
        dest='filter_inclusive_labels',
        help="Words in sample ID (map, episode, agent, frame) "
             "to filter sample files by.")
    return argparser.parse_args()

def dict_from_list(f, xs):
    g = lambda x: (f(x), x)
    return dict( util.map_to_list(g, xs) )

def main():
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    config = parse_arguments()
    group_creator = SampleGroupCreator(config)
    split_creator = CrossValidationSplitCreator(config)

    logging.info("Loading the data.")
    envs = []
    for file_path in config.data_files:
        try:
            loadpath = os.path.join(config.data_dir, file_path)
            with open(loadpath, 'rb') as f:
                env = dill.load(f, encoding='latin1')
                envs.append(env)
        except:
            logging.warning(f"Failed to load {file_path} from data directory")
            logging.warning(config.data_dir)

    env = None
    if len(envs) == 0:
        logging.warning("No data. Doing nothing.")
        return
    elif len(env) > 0:
        logging.info(f"Loaded {len(envs)} data payloads.")
        env = envs[0]
    for _env in envs[1:]:
        env.scenes.extend(_env.scenes)
    logging.info(f"Collected {len(env.scenes)} samples.")
    sample_dict = dict_from_list(lambda scene: scene.name, env.scenes)
    sample_ids = list(sample_dict.keys())
    logging.info(f"Creating group indices of samples.")
    groups = group_creator.make_groups(sample_ids)
    logging.info(f"Creating split indices of samples.")
    splits = split_creator.make_splits(groups)
    
    for idx, split in enumerate(splits[:config.n_splits]):
        split_idx = idx + 1
        logging.info(f"Making (train, val, test) split {split_idx}.")
        train_env = make_environment(env.name)
        val_env = make_environment(env.name)
        test_env = make_environment(env.name)
        train_env.scenes = util.map_to_list(sample_dict.get, split[0])
        val_env.scenes = util.map_to_list(sample_dict.get, split[1])
        test_env.scenes = util.map_to_list(sample_dict.get, split[2])

        logging.info(f"Augmenting scenes for split {split_idx}.")
        for scene in tqdm(train_env.scenes):
            scene.augmented = list()
            angles = np.arange(0, 360, 15)
            for angle in angles:
                scene.augmented.append(augment_scene(scene, angle))

        logging.info("Saving train set.")
        savepath = os.path.join(config.split_dir,
                "{label}_split{split_idx}_train.pkl".format(
                    split_idx=split_idx, label=config.label))
        with open(savepath, 'rb') as f:
            dill.dump(train_env, f, protocol=dill.HIGHEST_PROTOCOL)

        logging.info("Saving val set.")
        savepath = os.path.join(config.split_dir,
                "{label}_split{split_idx}_val.pkl".format(
                    split_idx=split_idx, label=config.label))
        with open(savepath, 'rb') as f:
            dill.dump(val_env, f, protocol=dill.HIGHEST_PROTOCOL)
        
        logging.info("Saving test set.")
        savepath = os.path.join(config.split_dir,
                "{label}_split{}_test_{}.pkl".format(
                    split_idx=split_idx, label=config.label))
        with open(savepath, 'rb') as f:
            dill.dump(test_env, f, protocol=dill.HIGHEST_PROTOCOL)
    
    logging.info("Done.")
