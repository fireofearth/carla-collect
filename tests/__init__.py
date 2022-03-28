import os
import enum

import carla
import utility as util
import carlautil

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_MAP = 'Town03'
SEED = 1

class LoopEnum(enum.Enum):
    """Indicator for toggling driving open/closed loop"""
    OPEN_LOOP = 0
    CLOSED_LOOP = 1

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

MODEL_SPEC_5 = util.AttrDict(
    path="experiments/nuScenes/models/20220326/models_26_Mar_2022_23_31_28_carla_v4-1_base_distmapV4_modfm_K15_ph8",
    desc="Base +MapV4-1 model with heading fix, rebalancing, PH=8, K=5 (trained on CARLA 0.9.13 all maps)"
)

MODEL_SPEC_6 = util.AttrDict(
    path="experiments/nuScenes/models/20220326/models_27_Mar_2022_04_52_55_carla_v4-2-1_base_distmapV4_modfm_K15_ph8",
    desc="Base +MapV4-1 model with heading fix, rebalancing, PH=8, K=5 (trained on CARLA 0.9.13 Town03)"
)

class ScenarioParameters(util.AttrDict):
    """Parameters for scenario running.
    Attributes
    ==========
    ego_spawn_idx : int
        Index of spawn point to place EV.
    other_spawn_ids : list of int
        Indices of spawn point to place OVs.
        The index of the i-th place designates where to spawn the i-th OV.
    other_routes : list of (list of str)
        The custom route for each OV using route commands. Applicable for
        CARLA versions 0.9.13 and above.
        The list at the i-th place designates the route of the i-th OV.
        Route commands are the strings: Void, Left, Right, Straight
        LaneFollow, ChangeLaneLeft, ChangeLaneRight, RoadEnd
    spawn_shifts : number or none
        spawn shifts for the vehicles j=1,2,... in
        `[ego_spawn_idx] + other_spawn_ids`.
        Value of `spawn_shifts[j]` is the distance
        from original spawn point to place vehicle j.
        Let `spawn_shifts[j] = None` to disable shifting.
    n_burn_interval : int
        Number of timesteps before starting motion planning.
    run_interval : int
        Number of steps to run motion planner.
        Only applicable to closed loop.
    controls : list of util.AttrDict
        Optional deterministic controls to apply to vehicle. Each control has
        attributes `interval` that specifies which frames to apply control,
        and `control` containing a carla.VehicleControl to apply to vehicle.
    goal : util.AttrDict
        Optional goal destination the motion planned vehicle should go to.
        By default the vehicle moves forwards.
        Not applicable to curved road segmented boundary constraints.
    turn_choices : list of int
        Indices of turns at each junction along the path from start_wp onwards.
    max_distance : number
        The maximum distance that EV can travel down the path.
    """

    def __init__(self,
            ego_spawn_idx=None,
            other_spawn_ids=[],
            other_routes=[],
            spawn_shifts=[],
            n_burn_interval=None,
            run_interval=None,
            controls=[],
            goal=None,
            turn_choices=[],
            max_distance=100,
            ignore_signs=True,
            ignore_lights=True,
            ignore_vehicles=True,
            auto_lane_change=False):
        super().__init__(
            ego_spawn_idx=ego_spawn_idx,
            other_spawn_ids=other_spawn_ids,
            other_routes=other_routes,
            spawn_shifts=spawn_shifts,
            n_burn_interval=n_burn_interval,
            run_interval=run_interval,
            controls=controls,
            goal=goal,
            turn_choices=turn_choices,
            max_distance=max_distance,
            ignore_signs=ignore_signs,
            ignore_lights=ignore_lights,
            ignore_vehicles=ignore_vehicles,
            auto_lane_change=auto_lane_change
        )


class CtrlParameters(util.AttrDict):
    """Parameters for MPC predictive control."""

    def __init__(self,
            n_predictions=100,
            prediction_horizon=8,
            control_horizon=8,
            step_horizon=1,
            n_coincide=1,
            random_mcc=False,
            loop_type=LoopEnum.OPEN_LOOP):
        super().__init__(
            n_predictions=n_predictions,
            prediction_horizon=prediction_horizon,
            control_horizon=control_horizon,
            step_horizon=step_horizon,
            n_coincide=n_coincide,
            random_mcc=random_mcc,
            loop_type=loop_type
        )


def shift_spawn_point(carla_map, k, spawn_shifts, spawn_point):
    try:
        spawn_shift = spawn_shifts[k]
        spawn_shift < 0
    except (TypeError, IndexError) as e:
        return spawn_point
    return carlautil.move_along_road(carla_map, spawn_point, spawn_shift)


def attach_camera_to_spectator(world, frame, sensor_tick="0.2"):
    os.makedirs(f"out/starting{frame}", exist_ok=True)
    blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    blueprint.set_attribute('image_size_x', '512')
    blueprint.set_attribute('image_size_y', '512')
    blueprint.set_attribute('fov', '80')
    blueprint.set_attribute('sensor_tick', sensor_tick)
    sensor = world.spawn_actor(
        blueprint, carla.Transform(), attach_to=world.get_spectator()
    )
    def take_picture(image):
        image.save_to_disk(
            f"out/starting{frame}/frame{image.frame}_spectator.png"
        )
    sensor.listen(take_picture)
    return sensor


