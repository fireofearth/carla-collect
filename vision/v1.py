import os
import logging
import time
import math
import random

import numpy as np
import scipy.spatial.transform

import carla
import utility as util
import utility.npu
import carlautil

from . import VisionException

class DataGenerator(object):
    # 8 maps
    MAP_NAMES = ["Town01", "Town02", "Town03", "Town04",
                 "Town05", "Town06", "Town07", "Town10HD"]

    def get_random_shuffled_spawn_points(self, carla_map):
        spawn_points = carla_map.get_spawn_points()
        random.shuffle(spawn_points)
        return spawn_points

    def load_worldmap(self, world_name):
        world = self.client.load_world(world_name)
        carla_map = world.get_map()
        weather = world.get_weather()
        spawn_points = self.get_random_shuffled_spawn_points(carla_map)
        return world, carla_map, weather, spawn_points

    def __init__(self, config):
        self.config = config
        self.n_scenes = self.config.n_scenes
        self.n_frames = self.config.n_frames
        self.show_demos = self.config.demo
        # delta : float
        #     Step size for synchronous mode.
        self.delta = 0.1
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(10.0)
        self.world, self.carla_map, self.original_weather, self.spawn_points \
                = self.load_worldmap(self.MAP_NAMES[0])
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.perturb_spawn_point = True

        self.near_r = 2.8

    def attach_camera_to_spectator(self, scene_idx):
        os.makedirs(f"out/snapshots/{self.carla_map.name}/scene{scene_idx}", exist_ok=True)
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", "256")
        blueprint.set_attribute("image_size_y", "256")
        blueprint.set_attribute("fov", "90")
        blueprint.set_attribute("sensor_tick", "0.1")
        sensor = self.world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self.world.get_spectator()
        )

        def save_snapshot(image):
            image.save_to_disk(f"out/snapshots/{self.carla_map.name}/scene{scene_idx}/frame{image.frame}.png")

        return sensor, save_snapshot

    def get_vehicle_blueprints(self):
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        return blueprints

    @staticmethod
    def demo_azimuthal_rotation(world, spectator, vehicle, iterations=300):
        theta = np.linspace(0, 2*math.pi, iterations)
        phi = math.pi / 4
        r = 4
        for jdx in range(iterations):
            transform = carlautil.spherical_to_camera_watcher_transform(
                        r, theta[jdx], phi, location=vehicle.get_location())
            transform = carlautil.strafe_transform(transform, right=0, above=0)
            spectator.set_transform(transform)
            if jdx == 0 or jdx == iterations - 1:
                for _ in range(50): world.wait_for_tick()
            else:
                world.wait_for_tick()
    
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
    
    def demo_night_lights(self, spectator, vehicle):
        """
        NOTE: for CARLA 9.11, not all cars have lights 
        """
        iterations = 300
        theta = np.linspace(-math.pi/2, (3/2)*math.pi, iterations)
        phi = (1/3)*math.pi
        r = 5
        original_lightstate = vehicle.get_light_state()
        vls = carla.VehicleLightState
        light_state = vls(vls.LowBeam | vls.Interior | vls.Reverse | vls.Position)
        vehicle.set_light_state(light_state)
        weather = self.world.get_weather()
        weather.sun_altitude_angle = -15
        self.world.set_weather(weather)
        
        for jdx in range(iterations):
            # set spectator orientation and position
            transform = carlautil.spherical_to_camera_watcher_transform(
                        r, theta[jdx], phi, location=vehicle.get_location())
            transform = carlautil.strafe_transform(transform, right=0, above=0)
            spectator.set_transform(transform)
            
            # wait for server before continuing
            if jdx == 0 or jdx == iterations - 1:
                for _ in range(50): self.world.wait_for_tick()
            else:
                self.world.wait_for_tick()

        vehicle.set_light_state(original_lightstate)

    def demo_all_lights(self, spectator, vehicle):
        iterations = 150
        theta = np.linspace(-math.pi/2, math.pi/2, iterations)
        phi = math.pi / 4
        r = 5
        sun_altitude_angles = np.linspace(1, -19, iterations)
        original_lightstate = vehicle.get_light_state()
        light_types = ["Position", "LowBeam", "HighBeam", "Brake",
                "RightBlinker", "LeftBlinker", "Reverse", "Fog",
                "Interior", "Special1", "Special2", "All"]
        for light_type in light_types:
            print(f"Demonstrating vehicle {light_type} lights.")
            vehicle.set_light_state(getattr(carla.VehicleLightState, light_type))
            for jdx in range(iterations):
                # set spectator orientation and position
                transform = carlautil.spherical_to_camera_watcher_transform(
                            r, theta[jdx], phi, location=vehicle.get_location())
                transform = carlautil.strafe_transform(transform, right=0, above=0)
                spectator.set_transform(transform)
                # set weather
                weather = self.world.get_weather()
                weather.sun_altitude_angle = sun_altitude_angles[jdx]
                self.world.set_weather(weather)
                # wait for server before continuing
                if jdx == 0 or jdx == iterations - 1:
                    for _ in range(50): self.world.wait_for_tick()
                else:
                    self.world.wait_for_tick()
        
        vehicle.set_light_state(original_lightstate)

    def __get_random_spawn_point(self, idx):
        n_spawn_points = len(self.spawn_points)
        if idx % n_spawn_points == 0 and idx > 0:
            self.spawn_points = self.get_random_shuffled_spawn_points(self.carla_map)
        return self.spawn_points[idx % n_spawn_points]
    
    def __perturb_spawn_point(self, spawn_point):
        spawn_shift = random.normalvariate(0.0, 5.0)
        perturb_spawn_point = carlautil.move_along_road(
            self.carla_map, spawn_point, spawn_shift
        )
        yaw = math.degrees(random.uniform(0, 2*math.pi))
        perturb_spawn_point.rotation = carla.Rotation(yaw=yaw)
        perturb_spawn_point.location += carla.Location(0, 0, 0.1)
        return perturb_spawn_point

    def __set_random_weather(self, vehicle):
        vls = carla.VehicleLightState
        weather = self.world.get_weather()
        altitude_angle = util.map_01_to_uv(-20, 90)(np.random.beta(5, 13))
        weather.sun_altitude_angle = altitude_angle
        deposits = util.map_01_to_uv(0, 60)(np.random.beta(0.4, 2.0))
        weather.precipitation_deposits = deposits
        cloudiness = util.map_01_to_uv(0, 100)(np.random.beta(0.4, 2.0))
        weather.cloudiness = cloudiness
        weather.wind_intensity = 50
        wetness = util.map_01_to_uv(0, 50)(np.random.beta(0.4, 2.0))
        weather.wetness = wetness
        self.world.set_weather(weather)
        if altitude_angle <= 0:
            light_state = vls(vls.LowBeam | vls.Interior | vls.Reverse | vls.Position)
            vehicle.set_light_state(light_state)

    def __set_random_frame(self, spectator, vehicle):
        r = random.uniform(self.near_r, 4.5)
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

    def run_scene(self, scene_idx, spawn_point):
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
            has_vehicle = False
            spawn_point = self.__get_random_spawn_point(scene_idx)
            for _ in range(10):
                _spawn_point = carlautil.carlacopy(spawn_point)
                _spawn_point = self.__perturb_spawn_point(_spawn_point)
                vehicle = self.world.try_spawn_actor(blueprint, _spawn_point)
                if vehicle:
                    has_vehicle = True
                    break
            if not has_vehicle:
                logging.warning(
                    f"Failed to spawn vehicle with perturb at {spawn_point}; "
                    "spawning vehicle without perturb."
                )
                vehicle = self.world.spawn_actor(blueprint, spawn_point)
            if self.show_demos:
                self.world.wait_for_tick()
                self.demo_azimuthal_rotation(self.world, spectator, vehicle)
                self.demo_polar_rotation(spectator, vehicle)
                self.demo_radius(spectator, vehicle)
                self.demo_shift(spectator, vehicle)
                self.demo_night_lights(spectator, vehicle)
                self.demo_all_lights(spectator, vehicle)
            else:
                self.world.tick()
                self.__set_random_weather(vehicle)
                self.__set_random_frame(spectator, vehicle)
                # wait for car jiggle to stop
                for _ in range(20): self.world.tick()
                self.__set_random_frame(spectator, vehicle)
                self.world.tick()
                camera, save_snapshot = self.attach_camera_to_spectator(scene_idx)
                camera.listen(save_snapshot)
                self.world.tick()
                for jdx in range(self.n_frames):
                    self.__set_random_frame(spectator, vehicle)
                    self.world.tick()
                camera.stop()
                self.world.tick()
        finally:
            if vehicle:
                vehicle.destroy()
            if camera:
                camera.destroy()
            if self.show_demos:
                self.world.set_weather(self.original_weather)
                self.world.wait_for_tick()
            else:
                self.world.tick()
            time.sleep(0.1) # wait to avoid crashing

    def run_scenes(self, n_scenes_per_map):
        logging.info(f"Getting {self.n_frames} frames, {n_scenes_per_map} scenes from map {self.carla_map.name}.")
        for idx in range(n_scenes_per_map):
            # set spawn point
            n_spawn_points = len(self.spawn_points)
            if idx % n_spawn_points == 0 and idx > 0:
                self.spawn_points = self.get_random_shuffled_spawn_points(self.carla_map)
            spawn_point = self.spawn_points[idx % n_spawn_points]
            if self.perturb_spawn_point:
                spawn_shift = random.normalvariate(0.0, 3.0)
                spawn_point = carlautil.move_along_road(
                    self.carla_map, spawn_point, spawn_shift
                )
                yaw = random.uniform(0, 2*math.pi)
                spawn_point.rotation = carla.Rotation(yaw=yaw)
                spawn_point.location += carla.Location(0, 0, 0.5)
            self.run_scene(idx, spawn_point)
        time.sleep(0.1) # wait to avoid crashing

    def run_synchronously(self, f):
        """Run in synchronous mode"""
        original_settings = None
        if self.show_demos:
            raise VisionException("Can't do demos synchronously")
        try:
            original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = self.delta
            settings.synchronous_mode = True
            self.traffic_manager.set_synchronous_mode(True)
            self.world.apply_settings(settings)
            logging.info(f"Set CARLA Server to synchronous mode.")
            f()
        finally:
            if original_settings:
                self.world.apply_settings(original_settings)
            self.world.wait_for_tick()
            logging.info(f"Set CARLA Server back to asynchronous mode.")
            time.sleep(0.1) # wait to avoid crashing

    def run(self):
        n_scenes_per_map = max(1, self.n_scenes // len(self.MAP_NAMES))
        for map_name in self.MAP_NAMES:
            if self.carla_map.name != map_name:
                self.world, self.carla_map, self.original_weather, self.spawn_points \
                        = self.load_worldmap(map_name)
                logging.info(f"Loaded map {self.carla_map.name}")
            if self.show_demos:
                self.run_scenes(1)
                break
            else:
                self.run_synchronously(lambda : self.run_scenes(n_scenes_per_map))
