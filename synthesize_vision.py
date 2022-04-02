import logging
import argparse
import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import carla
import utility as util
import carlautil

from vision.v4 import DataGenerator


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
        "--n-scenes",
        default=4000,
        type=int,
        help=(
            "Number of scenes to synthesize data. ",
            "The scenes are split among the available CARLA maps.",
        ),
    )
    argparser.add_argument(
        "--n-frames",
        default=10,
        type=int,
        help=(
            "Number of frames to attempt to capture for each scene. "
            "The RGB camera is unable to capture all the frames. "
            "In CARLA 0.9.11 Normally the RGB camera captures 10 frames out of 20."
            "This seems to be fixed in version 0.9.12."
        ),
    )
    argparser.add_argument(
        "--add-pedestrian",
        action="store_true",
        help=""
    )
    argparser.add_argument("--demo", action="store_true", help="")
    return argparser.parse_args()


def main():
    config = parse_arguments()
    log_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s: %(levelname)s: %(message)s", level=log_level
    )
    dg = DataGenerator(config)
    dg.run()


def visualize_rotation_order():
    """Visualize rotation order.
    Rotation uses the z-y'-x'' sequence (of intrinsic rotations).
    It is also known as the yaw, pitch, roll order.
    """
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    transform = carla_map.get_spawn_points()[0]
    transform.location += carla.Location(0, 0, 3.5)
    transform.rotation = carla.Rotation()
    spectator = world.get_spectator()
    spectator.set_transform(transform)
    pitch = 0.0
    yaw = 0.0
    roll = 0.0

    def move_pitch():
        nonlocal pitch
        for pitch in np.linspace(0, np.pi / 4, 20):
            world.wait_for_tick()
            transform.rotation = carla.Rotation(
                pitch=math.degrees(pitch),
                yaw=math.degrees(yaw),
                roll=math.degrees(roll),
            )
            spectator.set_transform(transform)

    def move_yaw():
        nonlocal yaw
        for yaw in np.linspace(0, np.pi / 2, 20):
            world.wait_for_tick()
            transform.rotation = carla.Rotation(
                pitch=math.degrees(pitch),
                yaw=math.degrees(yaw),
                roll=math.degrees(roll),
            )
            spectator.set_transform(transform)

    def move_roll():
        nonlocal roll
        for roll in np.linspace(0, np.pi / 4, 20):
            world.wait_for_tick()
            transform.rotation = carla.Rotation(
                pitch=math.degrees(pitch),
                yaw=math.degrees(yaw),
                roll=math.degrees(roll),
            )
            spectator.set_transform(transform)

    def demo_rot():
        movements = [move_pitch, move_yaw, move_roll]
        random.shuffle(movements)
        for f in movements:
            print(f)
            f()
        time.sleep(1)

    # move_roll()
    # move_yaw()
    # move_pitch()
    demo_rot()


def visualize_weather():
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    if carla_map.name != "Town03":
        world = client.load_world("Town03")
        carla_map = world.get_map()

    weather = world.get_weather()

    # float : Values range from 0 to 100, being 0 a clear sky and 100 one completely covered with clouds.
    weather.cloudiness = 0

    # float : Rain intensity values range from 0 to 100, being 0 none at all and 100 a heavy rain
    # NOTE: rain doesn't really work in OpenGL, or Vulkan
    weather.precipitation = 0

    # Determines the creation of puddles.
    # Values range from 0 to 100, being 0 none at all and 100 a road completely capped with water.
    # Puddles are created with static noise, meaning that they will always appear at the same locations.
    # NOTE: best to set this between 0-60
    weather.precipitation_deposits = 0

    # Controls the strenght of the wind with values from 0,
    # no wind at all, to 100, a strong wind.
    # The wind does affect rain direction and leaves from trees,
    # so this value is restricted to avoid animation issues.
    weather.wind_intensity = 0

    # Wetness intensity. It only affects the RGB camera sensor. Values range from 0 to 100
    # NOTE: best to set this between 0-50
    weather.wetness = 0


    # The azimuth angle of the sun. Values range from 0 to 360.
    # Zero is an origin point in a sphere determined by Unreal Engine.
    # NOTE: this just specifies which angle the sun rises/lowers.
    # Setting to another value breaks
    weather.sun_azimuth_angle = 0

    # Altitude angle of the sun. Values range from -90 to 90 corresponding to midnight and midday each.
    # NOTE: effects of setting angle below zero are the same (night)
    weather.sun_altitude_angle = -2

    # Fog concentration or thickness. It only affects the RGB camera sensor. Values range from 0 to 100.
    # NOTE: fog related settings do not work in OpenGL
    # NOTE: setting fog_density > 40 makes the sky turn black 
    # weather.fog_density = 0

    # Fog start distance. Values range from 0 to infinite (in meters).
    # weather.fog_distance = 0

    # Density of the fog (as in specific mass) from 0 to infinity. The bigger the value, the more dense and heavy it will be, and the fog will reach smaller heights. Corresponds to Fog Height Falloff in the UE docs.
    # If the value is 0, the fog will be lighter than air, and will cover the whole scene.
    # A value of 1 is approximately as dense as the air, and reaches normal-sized buildings.
    # For values greater than 5, the air will be so dense that it will be compressed on ground level.
    # weather.fog_falloff = 5

    # Controls interaction of light with large particles like pollen or air pollution resulting in a hazy sky with halos around the light sources.
    # When set to 0, there is no contribution.
    # NOTE: increases memory of simulation significantly
    # print("mie_scattering_scale", weather.mie_scattering_scale)

    # NOTE: set to < 0.1 or get weird light effects. Maps may have different scattering scales.
    # print("rayleigh_scattering_scale", weather.rayleigh_scattering_scale)

    world.set_weather(weather)


def visualize_sunshine():
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    altitude_angles = np.linspace(-90, 90, 180 + 1)
    for altitude_angle in altitude_angles:
        weather = world.get_weather()
        weather.sun_altitude_angle = altitude_angle
        world.set_weather(weather)
        world.wait_for_tick()


def visualize_wheel_turning_1():
    """This method shows that setting steer = 1.0 makes the vehicle
    turn to the max steering angle."""
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint = world.get_blueprint_library().find("vehicle.audi.a2")
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = None
    try:
        vehicle = world.spawn_actor(blueprint, spawn_point)
        world.wait_for_tick()
        phys_control = vehicle.get_physics_control()
        wheel = phys_control.wheels[1]

        transform = carlautil.spherical_to_camera_watcher_transform(
                3, math.pi, math.pi*(1/6),
                location=vehicle.get_location())
        world.get_spectator().set_transform(transform)

        # Q1: does applying steering control once persist across many ticks?
        world.wait_for_tick()
        vehicle.apply_control(carla.VehicleControl(steer=1.0))
        world.wait_for_tick()
        phys_control = vehicle.get_physics_control()
        wheel = phys_control.wheels[1]

        location = vehicle.get_location()
        transform = vehicle.get_transform()
        end = location + 5*transform.get_forward_vector()
        world.debug.draw_line(location, end,
                thickness=0.1, life_time=15.0)
        
        # Q2: does steer=1.0 correspond with wheel.max_steer_angle?
        location = vehicle.get_location()
        transform = vehicle.get_transform()
        theta = math.radians(90 - wheel.max_steer_angle)
        direction = math.sin(theta)*transform.get_forward_vector() \
                + math.cos(theta)*transform.get_right_vector()
        end = location + 5*direction
        world.debug.draw_line(location, end,
                thickness=0.1, life_time=15.0)

        # A1: yes it does.
        for _ in range(800): world.wait_for_tick()

    finally:
        if vehicle:
            vehicle.destroy()

def visualize_wheel_turning_2():
    """
    Q: is the steering immediate?
    A: no it takes time. Small steering rate (i.e. setting steer=0.2) is fast.
    The larger the change in steering, the longer it takes the wheels to change from
    the current angle to the intended angle.
    https://github.com/carla-simulator/carla/issues/4655
    """
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint = world.get_blueprint_library().find("vehicle.audi.a2")
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = None
    try:
        vehicle = world.spawn_actor(blueprint, spawn_point)
        world.wait_for_tick()
        phys_control = vehicle.get_physics_control()
        wheel = phys_control.wheels[1]

        transform = carlautil.spherical_to_camera_watcher_transform(
                3, math.pi, math.pi*(1/6),
                location=vehicle.get_location())
        world.get_spectator().set_transform(transform)
        for _ in range(100): world.wait_for_tick()

        world.wait_for_tick()
        vehicle.apply_control(carla.VehicleControl(steer=0.2))
        world.wait_for_tick()
        time.sleep(0.5)
        world.wait_for_tick()
        vehicle.apply_control(carla.VehicleControl(steer=0.0))
        world.wait_for_tick()
        time.sleep(0.5)
        world.wait_for_tick()
        vehicle.apply_control(carla.VehicleControl(steer=0.5))
        world.wait_for_tick()
        time.sleep(0.5)
        world.wait_for_tick()
        vehicle.apply_control(carla.VehicleControl(steer=0.0))
        world.wait_for_tick()
        time.sleep(0.5)
        world.wait_for_tick()
        vehicle.apply_control(carla.VehicleControl(steer=1.0))
        world.wait_for_tick()
        time.sleep(0.5)


        for _ in range(200): world.wait_for_tick()
    finally:
        if vehicle:
            vehicle.destroy()


def visualize_wheel_turning_3():
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint = world.get_blueprint_library().find("vehicle.audi.a2")
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = None
    vwl = carla.VehicleWheelLocation

    try:
        vehicle = world.spawn_actor(blueprint, spawn_point)
        for _ in range(20): world.wait_for_tick()
        transform = carlautil.spherical_to_camera_watcher_transform(
                3, math.pi, math.pi*(1/6),
                location=vehicle.get_location())
        world.get_spectator().set_transform(transform)
        for _ in range(20): world.wait_for_tick()

        fl_angles = dict()
        fr_angles = dict()
        def plot(snapshot):

            fl_angle = vehicle.get_wheel_steer_angle(vwl.FL_Wheel)
            fr_angle = vehicle.get_wheel_steer_angle(vwl.FR_Wheel)
            fl_angles[snapshot.timestamp.elapsed_seconds] = fl_angle
            fr_angles[snapshot.timestamp.elapsed_seconds] = fr_angle

        def derivative(x, y):
            assert len(x) == len(y)
            dy = [None]*(len(x) - 1)
            for i in range(len(x) - 1):
                dy[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            assert len(x[:-1]) == len(dy)
            return x[:-1], dy

        ## Block 1
        callback_id = world.on_tick(plot)
        """get_wheel_steer_angle() returns angle in degrees between
        45 and 70 deg so the actual turning is more like 57.5 deg.
        The wheel on the side the car is turning turns a higher angle.
        Wheels reach max turns from 0 in 0.2 seconds so 287.5 deg/s."""

        steering_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # steering_choices = [0.6]
        for steer in steering_choices:
            for _ in range(6): world.wait_for_tick()
            time.sleep(0.8)
            for _ in range(6): world.wait_for_tick()
            vehicle.apply_control(carla.VehicleControl(steer=steer))
            for _ in range(6): world.wait_for_tick()
            time.sleep(0.8)
            for _ in range(6): world.wait_for_tick()
            vehicle.apply_control(carla.VehicleControl(steer=0))

        for _ in range(6): world.wait_for_tick()
        time.sleep(0.8)
        world.remove_on_tick(callback_id)
        world.wait_for_tick()
        fig, axes = plt.subplots(1, 2, figsize=(20,10))
        _time, _angles = util.unzip(fl_angles.items())
        axes[0].plot(_time, _angles, "-bo", label="FL_Wheel")
        _time, _dangles = derivative(_time, _angles)
        axes[1].plot(_time, _dangles, "-bo", label="FL_Wheel")

        _time, _angles = util.unzip(fr_angles.items())
        axes[0].plot(_time, _angles, "-ro", label="FR_Wheel")
        _time, _dangles = derivative(_time, _angles)
        axes[1].plot(_time, _dangles, "-ro", label="FR_Wheel")

        axes[0].legend()

        fig.savefig("out/steer")
        # plt.show()

    finally:
        if vehicle:
            vehicle.destroy()
            vehicle = None


def inspect_vehicle():
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint = world.get_blueprint_library().find("vehicle.audi.a2")
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = None
    try:
        vehicle = world.spawn_actor(blueprint, spawn_point)
        world.wait_for_tick()
        transform = carlautil.spherical_to_camera_watcher_transform(
                3, math.pi, math.pi*(1/6),
                location=vehicle.get_location())
        world.get_spectator().set_transform(transform)
        world.wait_for_tick()
        def get_forward_vector():
            """ The vehicle's forward vector is a unit vector """
            f = vehicle.get_transform().get_forward_vector()
            f = np.array([f.x, f.y, f.z])
            print( np.linalg.norm(f) )
        def get_bounding_box():
            bb = vehicle.bounding_box
            e = bb.extent
            print(np.array([e.x, e.y, e.z])*2)
        # get_bounding_box()

    finally:
        if vehicle:
            vehicle.destroy()


def jump_to_transform():
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    # carla_map = world.get_map()
    transform = carla.Transform(
        carla.Location(x=88.61,y=90.58,z=0.3),
        carla.Rotation(pitch=0, yaw=90, roll=0)
    )
    world.get_spectator().set_transform(transform)


def visualize_pedestrian():
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(5.0)
    world = client.get_world()
    carla_map = world.get_map()

    walker_blueprints = world.get_blueprint_library().filter("walker.*")
    spawn_point = carla_map.get_spawn_points()[0]
    pedestrian = None
    spectator = world.get_spectator()

    for blueprint in walker_blueprints:
        try:
            print("Showing pedestrian", blueprint)
            pedestrian = world.spawn_actor(blueprint, spawn_point)
            world.wait_for_tick()
            iterations = 40
            theta = np.linspace(0, 1*math.pi, iterations)
            phi = (5/6 / 2)*math.pi
            r = 2
            for jdx in range(iterations):
                transform = carlautil.spherical_to_camera_watcher_transform(
                        r, theta[jdx], phi, location=pedestrian.get_location())
                # transform = carlautil.strafe_transform(
                #         transform, right=0, above=0)
                spectator.set_transform(transform)
                world.wait_for_tick()
        finally:
            if pedestrian:
                pedestrian.destroy()


def visualize_vehicles_in_the_dark():
    """Cars with lights:

    audi.tt
    mercedes.sprinter
    dodge.charger_police
    mercedes.coupe_2020
    dodge.charger_police_2020
    mini.cooper_s_2021
    bh crossbike
    lincoln.mkz_2020
    lincoln.mkz_2017
    ford.mustnge
    tesla ybertruck
    ford ambulance
    nissan patrol_2021
    yamaha yzf
    chevrolet impala
    volkswagen t2
    diamondback century
    tesla model3
    carlamotos firetruck
    audi etron
    dodge charger_2020
    gazelle omafiets
    harley-deavidson low_rider
    

    """
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(5.0)
    world = client.get_world()
    carla_map = world.get_map()
    weather = world.get_weather()
    weather.sun_altitude_angle = -15
    world.set_weather(weather)
    vehicle_blueprints = world.get_blueprint_library().filter("vehicle.*")
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = None
    pedestrian = None
    spectator = world.get_spectator()

    for blueprint in vehicle_blueprints:
        try:
            print("Showing vehicle", blueprint)
            vehicle = world.spawn_actor(blueprint, spawn_point)
            world.wait_for_tick()
            vls = carla.VehicleLightState
            light_state = vls(vls.LowBeam | vls.Interior | vls.Reverse | vls.Position)
            vehicle.set_light_state(light_state)

            iterations = 100
            theta = np.linspace(0, 1*math.pi, iterations)
            phi = math.pi / 4
            r = 5
            for jdx in range(iterations):
                transform = carlautil.spherical_to_camera_watcher_transform(
                        r, theta[jdx], phi, location=vehicle.get_location())
                # transform = carlautil.strafe_transform(
                #         transform, right=0, above=0)
                spectator.set_transform(transform)
                world.wait_for_tick()
        finally:
            if vehicle:
                vehicle.destroy()

def inspect_entity_dimensions():
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(5.0)
    world = client.get_world()
    carla_map = world.get_map()
    spawn_point = carla_map.get_spawn_points()[0]
    # vehicle blueprints
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    blueprints = [x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
    blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
    blueprints = [x for x in blueprints if not x.id.endswith('t2')]
    # walker blueprints
    walker_blueprints = world.get_blueprint_library().filter("walker.*")
    vehicle = None
    pedestrian = None
    spectator = world.get_spectator()
    logging = util.AttrDict(
        vehicle=util.AttrDict(
            xs=[],
            d2s=[],
            d3s=[]
        ),
        pedestrian=util.AttrDict(
            xs=[],
            d2s=[],
            d3s=[]
        )
    )
    for blueprint in blueprints:
        try:
            vehicle = world.spawn_actor(blueprint, spawn_point)
            world.wait_for_tick()
            transform = carlautil.spherical_to_camera_watcher_transform(
                    5, math.pi, math.pi*(1/6),
                    pin=spawn_point.location)
            spectator.set_transform(transform)
            world.wait_for_tick()
            x = carlautil.actor_to_bbox_ndarray(vehicle)
            d2 = np.linalg.norm(x[:2] / 2)
            d3 = np.linalg.norm(x / 2)
            logging.vehicle.xs.append(x)
            logging.vehicle.d2s.append(d2)
            logging.vehicle.d3s.append(d3)
            print(blueprint.id)
            print(x)
            time.sleep(0.1)
        finally:
            if vehicle:
                vehicle.destroy()
            vehicle = None
    world.wait_for_tick()
    for blueprint in walker_blueprints:
        try:
            pedestrian = world.spawn_actor(blueprint, spawn_point)
            world.wait_for_tick()
            transform = carlautil.spherical_to_camera_watcher_transform(
                    5, math.pi, math.pi*(1/6),
                    pin=spawn_point.location)
            spectator.set_transform(transform)
            world.wait_for_tick()
            x = carlautil.actor_to_bbox_ndarray(pedestrian)
            d2 = np.linalg.norm(x[:2] / 2)
            d3 = np.linalg.norm(x / 2)
            logging.pedestrian.xs.append(x)
            logging.pedestrian.d2s.append(d2)
            logging.pedestrian.d3s.append(d3)
        finally:
            if pedestrian:
                pedestrian.destroy()
            pedestrian = None

    for k, entity in logging.items():
        entity.xs = np.stack(entity.xs)
        entity.d2s = np.stack(entity.d2s)
        entity.d3s = np.stack(entity.d3s)
    print(f"Vehicle dimensions { np.max(logging.vehicle.xs, 0) }")
    print(f"    max 2D distance from origin { np.max(logging.vehicle.d2s) }")
    print(f"    max 3D distance from origin { np.max(logging.vehicle.d3s) }")

    print(f"Pedestrian dimensions { np.mean(logging.pedestrian.xs, 0) }")
    print(f"    max 2D distance from origin { np.max(logging.pedestrian.d2s) }")
    print(f"    max 3D distance from origin { np.max(logging.pedestrian.d3s) }")


def visualize_distances():
    config = parse_arguments()
    client = carla.Client(config.host, config.port)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    blueprints = [x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
    blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
    blueprints = [x for x in blueprints if not x.id.endswith('t2')]
    # blueprint = world.get_blueprint_library().find("vehicle.audi.a2")
    blueprint = random.choice(blueprints)
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = None
    try:
        vehicle = world.spawn_actor(blueprint, spawn_point)
        transform = carlautil.spherical_to_camera_watcher_transform(
                3, math.pi, math.pi*(1/6),
                pin=spawn_point.location)
        world.get_spectator().set_transform(transform)

        thickness = 0.2
        color = carla.Color(r=255, g=0, b=0, a=100)
        life_time = 6
        loc = carlautil.to_location_ndarray(spawn_point.location)
        scale = 2.
        z = loc[2] + 0.5
        box = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        box = (box*scale) + loc[:2]
        world.debug.draw_line(
            carla.Location(box[0, 0], box[0, 1], z),
            carla.Location(box[1, 0], box[1, 1], z),
            thickness=thickness, color=color, life_time=life_time)
        world.debug.draw_line(
            carla.Location(box[1, 0], box[1, 1], z),
            carla.Location(box[2, 0], box[2, 1], z),
            thickness=thickness, color=color, life_time=life_time)
        world.debug.draw_line(
            carla.Location(box[2, 0], box[2, 1], z),
            carla.Location(box[3, 0], box[3, 1], z),
            thickness=thickness, color=color, life_time=life_time)
        world.debug.draw_line(
            carla.Location(box[3, 0], box[3, 1], z),
            carla.Location(box[0, 0], box[0, 1], z),
            thickness=thickness, color=color, life_time=life_time)
        time.sleep(life_time)
        world.wait_for_tick()

    finally:
        if vehicle:
            vehicle.destroy()


if __name__ == "__main__":
    # visualize_rotation_order()
    # visualize_sunshine()
    # visualize_weather()
    # visualize_wheel_turning_2()
    # visualize_wheel_turning_3()
    # inspect_vehicle()
    # visualize_vehicles_in_the_dark()
    # jump_to_transform()
    inspect_entity_dimensions()
    # visualize_distances()
    # main()
