import pprint 
pp = pprint.PrettyPrinter(indent=4)

import pytest

import utility as util
import utility as util

from collect.generate.label import carla_id_maker
from collect.generate.dataset import SampleGroupCreator, CrossValidationSplitCreator

MOCK_IDS = [
    'Town01/ep001/agent001/frame00001000',
    'Town01/ep001/agent001/frame00001020',
    'Town01/ep001/agent002/frame00001000',
    'Town01/ep001/agent002/frame00001020',
    'Town01/ep001/agent003/frame00001000',
    'Town01/ep001/agent004/frame00001000',

    'Town01/ep002/agent001/frame00001000',
    'Town01/ep002/agent001/frame00001020',
    'Town01/ep002/agent002/frame00001000',
    'Town01/ep002/agent003/frame00001000',
    'Town01/ep002/agent003/frame00001020',
    'Town01/ep002/agent004/frame00001000',

    'Town02/ep003/agent001/frame00001000',
    'Town02/ep003/agent002/frame00001000',
    'Town02/ep003/agent002/frame00001020',
    'Town02/ep003/agent003/frame00001000',
    'Town02/ep003/agent004/frame00001000',
    'Town02/ep003/agent004/frame00001020',

    'Town02/ep004/agent001/frame00001000',
    'Town02/ep004/agent002/frame00001000',
    'Town02/ep004/agent003/frame00001000',
    'Town02/ep004/agent003/frame00001020',
    'Town02/ep004/agent004/frame00001000',
    'Town02/ep004/agent004/frame00001020',
]

def test_group_split():
    """Test by manual inspection"""
    config = util.AttrDict(n_groups=4)
    group_creator = SampleGroupCreator(config)
    split_creator = CrossValidationSplitCreator(config)
    sample_ids = MOCK_IDS
    groups = group_creator.make_groups(sample_ids)
    splits = split_creator.make_splits(groups)
    split = splits[0]
    print()
    pp.pprint(groups)
    pp.pprint(split)
