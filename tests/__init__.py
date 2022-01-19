import utility as util

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_MAP = 'Town03'
SEED = 1

MODEL_SPEC_1 = util.AttrDict(
        path="experiments/nuScenes/models/20210621/models_19_Mar_2021_22_14_19_int_ee_me_ph8",
        desc="Base +Dynamics, Map off-the-shelf model trained on NuScenes")

MODEL_SPEC_2 = util.AttrDict(
        path="experiments/nuScenes/models/models_20_Jul_2021_11_48_11_carla_v3_0_1_base_distmap_ph8",
        desc="Base +Map model w/ heading fix trained on small CARLA synthesized")

MODEL_SPEC_3 = util.AttrDict(
        path="experiments/nuScenes/models/20210803/models_03_Aug_2021_13_42_51_carla_v3-1-1_base_distmapV4_ph8",
        desc="Base +MapV4-1 model with heading fix, PH=8, K=25 "
             "(trained on smaller carla v3-1-1 dataset)")

MODEL_SPEC_4 = util.AttrDict(
        path="experiments/nuScenes/models/20210816/models_17_Aug_2021_13_25_38_carla_v3-1-2_base_distmapV4_modfm_K15_ph8",
        desc="Base +MapV4 model with heading fix, PH=8, K=15 "
             "(trained on carla v3-1-2 dataset)")
