"""Module constructing generic datasets no specific to any model architecture.
Does dataset grouping, shuffling, and constructs validation split."""

# Built-in libraries
import os
import json
import logging
import glob

# PyPI libraries
import numpy as np

# Local libararies
import utility as util

# Modules
from ..label import carla_id_maker

logger = logging.getLogger(__name__)

class CrossValidationSplitCreator(object):
    """Make cross validation splits from groups."""

    def __init__(self, config):

        self.config = config

    @staticmethod
    def __gen_splits(n):
            """Generator of group indices for (train, val, test) set.
            Only yields n-1 of the possible index combinations"""
            v = util.range_to_list(n)
            for idx in range(0, len(v) - 1):
                yield tuple(v[:idx] + v[idx + 2:]), (idx,), (idx + 1,)

    def make_splits(self, groups):
        """Make cross validation splits from groups.
        
        Parameters
        ==========
        groups : dict of int: (list of str)
            Groups of sample IDs.

        Returns
        =======
        list of (dict of (int: list of str))
            The splits. The splits are shuffled.
        """
        n_groups = len(groups)
        splits = []
        for train, val, test in self.__gen_splits(n_groups):
            split = {
                    0: util.merge_list_of_list([groups[idx] for idx in train]),
                    1: util.merge_list_of_list([groups[idx] for idx in val]),
                    2: util.merge_list_of_list([groups[idx] for idx in test])}
            util.shuffle_nested_dict_of_list(split)
            splits.append(split)
        return splits

class SampleGroupCreator(object):
    """Make groups from samples."""

    def __init__(self, config):

        self.config = config

        if config.n_groups < 3:
            raise ValueError(f"n_groups={config.n_groups} is too small!")
        # n_groups : int
        #   The number of groups to create.
        self.n_groups = config.n_groups
        # filter_inclusive_labels : dict of (str: list)
        #   Word to check and list of values to filter in.
        self.filter_inclusive_labels = getattr(
                config, 'filter_inclusive_labels', None)

    def make_groups(self, sample_ids):
        """Make groups like so:

        1. agent in an episode is not represented in more than one group.
        2. maps are represented evenly across all groups.
        3. groups are roughly the same size and is shuffled

        Parameters
        ==========
        bulk : dict of (str: any)
            The indexed objects to create groups from.

        Returns
        =======
        dict of int: (list of str)
            Sample IDs grouped into n_groups groups.
            These IDs are not shuffled.
        """
        groups = { }
        for idx in range(self.n_groups):
            groups[idx] = [ ]
        
        if self.filter_inclusive_labels:
            carla_id_maker.filter_ids(sample_ids, self.filter_inclusive_labels)
        
        nested_groups, labels = carla_id_maker.group_ids(sample_ids, ['map', 'episode', 'agent'])
        group_idx = 0
        for map_name in labels['map']:
            for episode_id in labels['episode']:
                for agent_id in labels['agent']:
                    # print(f"{map_name}/{episode_id}/{agent_id} to group {group_idx}")
                    groups[group_idx].extend(nested_groups[map_name][episode_id][agent_id])
                    group_idx = (group_idx + 1) % self.n_groups
        
        util.shuffle_nested_dict_of_list(groups)
        for idx, group in groups.items():
            logger.info(f"group {idx} has {len(group)} samples.")

        return groups
