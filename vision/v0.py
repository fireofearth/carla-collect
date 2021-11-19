import os
import time
import argparse
import copy
import math
import random

import numpy as np
import scipy.spatial.transform

import carla
import carlautil


def attach_camera_to_spectator(world):
    os.makedirs(f"out/snapshots", exist_ok=True)
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", "256")
    blueprint.set_attribute("image_size_y", "256")
    blueprint.set_attribute("fov", "90")
    # blueprint.set_attribute("sensor_tick", "0.5")
    sensor = world.spawn_actor(
        blueprint, carla.Transform(), attach_to=world.get_spectator()
    )

    def save_snapshot(image):
        image.save_to_disk(f"out/snapshots/frame{image.frame}.png")

    return sensor, save_snapshot

class DataGenerator(object):
    # MAP_NAMES = ["Town01", "Town02", "Town03", "Town04",
    #              "Town05", "Town06", "Town07", "Town10HD"]
    MAP_NAMES = ["Town03"]

    def load_worldmap(self, world_name):
        world = self.client.load_world(world_name)
        carla_map = world.get_map()
        return world, carla_map

    def __init__(self, config):
        self.config = config
        self.n_scenes = self.config.n_scenes
        self.n_frames = self.config.n_frames
        self.show_demos = self.config.demo
        self.delta = 0.1
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(10.0)
        self.world, self.carla_map = self.load_worldmap(self.MAP_NAMES[0])
        self.perturb_spawn_point = True

        self.near_r = 3

    def get_random_shuffled_spawn_points(self):
        spawn_points = self.carla_map.get_spawn_points()
        random.shuffle(spawn_points)
        return spawn_points

    def get_vehicle_blueprints(self):
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        return blueprints

    def demo_azimuthal_rotation(self, spectator, vehicle):
        iterations = 300
        theta = np.linspace(0, 2*math.pi, iterations)
        phi = math.pi / 4
        r = 4
        for jdx in range(iterations):
            transform = carlautil.spherical_to_camera_watcher_transform(
                        r, theta[jdx], phi, location=vehicle.get_location())
            transform = carlautil.strafe_transform(transform, right=0, above=0)
            spectator.set_transform(transform)
            if jdx == 0 or jdx == iterations - 1:
                for _ in range(50): self.world.wait_for_tick()
            else:
                self.world.wait_for_tick()
    
    def demo_polar_rotation(self, spectator, vehicle):
        iterations = 200
        theta = math.pi / 3
        phi = np.linspace((5/14)*math.pi, (1/6)*math.pi, iterations)
        r = 4
        for jdx in range(iterations):
            transform = carlautil.spherical_to_camera_watcher_transform(
                        r, theta, phi[jdx], location=vehicle.get_location())
            transform = carlautil.strafe_transform(transform, right=0, above=0)
            spectator.set_transform(transform)
            if jdx == 0 or jdx == iterations - 1:
                for _ in range(50): self.world.wait_for_tick()
            else:
                self.world.wait_for_tick()
    
    def demo_radius(self, spectator, vehicle):
        iterations = 100
        theta = math.pi / 2
        phi = math.pi / 4
        r = np.linspace(self.near_r, 5, iterations)
        for jdx in range(iterations):
            transform = carlautil.spherical_to_camera_watcher_transform(
                        r[jdx], theta, phi, location=vehicle.get_location())
            transform = carlautil.strafe_transform(transform, right=0, above=0)
            spectator.set_transform(transform)
            if jdx == 0 or jdx == iterations - 1:
                for _ in range(50): self.world.wait_for_tick()
            else:
                self.world.wait_for_tick()
    
    def demo_shift(self, spectator, vehicle):
        iterations = 100
        theta = math.pi / 2
        phi = math.pi / 4
        r = np.linspace(self.near_r, 5, iterations)
        near_frac = vehicle.bounding_box.extent.x
        ratio_f = near_frac / self.near_r
        for jdx in range(iterations):
            transform = carlautil.spherical_to_camera_watcher_transform(
                        r[jdx], theta, phi, location=vehicle.get_location())
            transform = carlautil.strafe_transform(transform,
                right=r[jdx]*ratio_f - near_frac,
                above=r[jdx]*ratio_f - near_frac)
            spectator.set_transform(transform)
            if jdx == 0 or jdx == iterations - 1:
                for _ in range(50): self.world.wait_for_tick()
            else:
                self.world.wait_for_tick()

    def __set_random_frame(self, spectator, vehicle):
        r = random.uniform(self.near_r, 5)
        theta = random.uniform(0, 2*math.pi)
        phi = random.uniform((5/14)*math.pi, (1/6)*math.pi)
        near_frac = vehicle.bounding_box.extent.x
        ratio_f = near_frac / self.near_r
        strafe_amt = r*ratio_f - near_frac
        right_strafe = random.uniform(-strafe_amt, strafe_amt)
        above_strafe = random.uniform(-strafe_amt, strafe_amt)

        transform = carlautil.spherical_to_camera_watcher_transform(
                r, theta, phi, location=vehicle.get_location())
        transform = carlautil.strafe_transform(transform,
                right=right_strafe, above=above_strafe)
        spectator.set_transform(transform)

    def run_scene(self, spawn_point):
        """Sample frames from a scene where the car is placed on some spawn point.
        """
        vehicle = None
        pedestrian = None
        camera = None
        spectator = self.world.get_spectator()

        # set vehicle blue prints
        blueprints = self.get_vehicle_blueprints()
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)

        try:
            # create vehicle and do things with it
            vehicle = self.world.spawn_actor(blueprint, spawn_point)
            if self.show_demos:
                self.world.wait_for_tick()
                self.demo_azimuthal_rotation(spectator, vehicle)
                self.demo_polar_rotation(spectator, vehicle)
                self.demo_radius(spectator, vehicle)
                self.demo_shift(spectator, vehicle)
            else:
                self.world.wait_for_tick()
                self.__set_random_frame(spectator, vehicle)
                # wait for car jiggle to stop
                for _ in range(60): self.world.wait_for_tick()
                self.__set_random_frame(spectator, vehicle)
                self.world.wait_for_tick()
                camera, save_snapshot = attach_camera_to_spectator(self.world)
                camera.listen(save_snapshot)
                self.world.wait_for_tick()
                for jdx in range(self.n_frames):
                    self.__set_random_frame(spectator, vehicle)
                    self.world.wait_for_tick()
                camera.stop()
                for _ in range(10): self.world.wait_for_tick()
        finally:
            if vehicle:
                vehicle.destroy()
            if camera:
                camera.destroy()
            time.sleep(0.1) # wait to avoid crashing

    def run(self):
        n_scenes_per_map = max(1, self.n_scenes // len(self.MAP_NAMES))
        for map_name in self.MAP_NAMES:
            if self.carla_map.name != map_name:
                self.world, self.carla_map = self.load_worldmap(map_name)
            for idx in range(n_scenes_per_map):

                # set spawn point
                spawn_points = self.get_random_shuffled_spawn_points()
                n_spawn_points = len(spawn_points)
                if idx % n_spawn_points == 0 and idx > 0:
                    spawn_points = self.get_random_shuffled_spawn_points()
                spawn_point = spawn_points[idx % n_spawn_points]
                if self.perturb_spawn_point:
                    spawn_shift = random.normalvariate(0.0, 10.0)
                    spawn_point = carlautil.move_along_road(
                        self.carla_map, spawn_point, spawn_shift
                    )
                    yaw = random.uniform(0, 2*math.pi)
                    spawn_point.rotation = carla.Rotation(yaw=yaw)
                    spawn_point.location += carla.Location(0, 0, 0.5)
                self.run_scene(spawn_point)


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
    argparser.add_argument("--n-scenes", default=1, type=int, help="")
    argparser.add_argument("--n-frames", default=10, type=int, help="")
    argparser.add_argument("--demo", action="store_true", help="")
    return argparser.parse_args()


def main():
    config = parse_arguments()
    dg = DataGenerator(config)
    dg.run()


def simple():
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


if __name__ == "__main__":
    # simple()
    main()
