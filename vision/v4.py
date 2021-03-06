"""2 Options for synthesizing images:

 - cars. Improvements from v3: vehicle doesn't roll when it is on a slope. 
 - cars and pedestrians.
 
"""
import os
import logging
import time
import math
import random
import functools
import json

import numpy as np
import shapely
import shapely.geometry
import scipy.spatial.transform

import carla
import utility as util
import utility.npu
import carlautil

from . import VisionException

class DemoMixin(object):
    @staticmethod
    def demo_azimuthal_rotation(world, spectator, vehicle, iterations=300):
        theta = np.linspace(0, 2*math.pi, iterations)
        phi = math.pi / 4
        r = 4
        for jdx in range(iterations):
            transform = carlautil.spherical_to_camera_watcher_transform(
                        r, theta[jdx], phi, pin=vehicle.get_location())
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
                        r, theta, phi[jdx], pin=vehicle.get_location())
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
        r = np.linspace(*self.range_r, iterations)
        for jdx in range(iterations):
            transform = carlautil.spherical_to_camera_watcher_transform(
                        r[jdx], theta, phi, pin=vehicle.get_location())
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
        r = np.linspace(*self.range_r, iterations)
        near_frac = vehicle.bounding_box.extent.x
        ratio_f = near_frac / self.range_r[0]
        for jdx in range(iterations):
            transform = carlautil.spherical_to_camera_watcher_transform(
                        r[jdx], theta, phi, pin=vehicle.get_location())
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
        NOTE: for CARLA 0.9.11/12, not all cars have lights 
        """
        iterations = 300
        theta = np.linspace(-math.pi/2, (3/2)*math.pi, iterations)
        phi = (1/3)*math.pi
        r = sum(self.range_r) / 2.
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
                        r, theta[jdx], phi, pin=vehicle.get_location())
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
        r = sum(self.range_r) / 2.
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
                            r, theta[jdx], phi, pin=vehicle.get_location())
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

class DataGenerator(DemoMixin):
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
    
    # @staticmethod
    # def augment_arguments(argparser):
    #     """Add arguments to argparse.ArgumentParser"""
    #     argparser.add_argument(
    #         "--range-r", nargs='+', type=float, default=[7.0, 8.0]
    #     )
    #     argparser.add_argument(
    #         "--range-theta", nargs='+', type=float, default=[0, 2*math.pi]
    #     )
        
    def __init__(self, config):
        self.config = config
        self.debug = self.config.debug
        self.out_path = "out/snapshots"
        # add_pedestrian : bool
        #   Whether to place a pedestrian in the scene.
        self.add_pedestrian = self.config.add_pedestrian
        # n_scenes : int
        #   Number of scenes. Scenes are split between existing maps.
        #   The camera is readjusted for each frame in while keeping the scene unchanged.
        self.n_scenes = self.config.n_scenes
        # n_frames : int
        #   Number of frames to capture per scene.
        self.n_frames = self.config.n_frames
        self.show_demos = self.config.demo
        # delta : float
        #   Step size for synchronous mode.
        self.delta = 0.1
        self.client = carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(10.0)
        self.world, self.carla_map, self.original_weather, self.spawn_points \
                = self.load_worldmap(self.MAP_NAMES[0])
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.perturb_spawn_point = True
        # range_r : list of float
        #   Near and far distance range from camera to placement center.
        self.range_r = [7.0, 8.0]
        # self.range_r = [7.0, 7.0]
        # range_theta : list of float
        #   Range of camera azimuth from placement center.
        self.range_theta = [0, 2*math.pi]
        # self.range_theta = [0, 0]
        # range_phi : list of float
        #   Range of camera altitude from placement center.
        self.range_phi = [(5/14)*math.pi, (1/6)*math.pi]
        # self.range_phi = [(5/14)*math.pi, (5/14)*math.pi]
        # max_shift : float
        #   Range of x and y shifts of foreground objects from center of camera view.
        if self.add_pedestrian:
            self.max_shift =  2.0
        else:
            # self.max_shift = 0.5
            self.max_shift = 2.0
        # ped_halfwidth : size of pedestrian bounding box
        self.ped_halfwidth = 0.5
        # camera_attributes : attributes to set when creating camera
        self.camera_attributes = util.AttrDict(
            image_size_x="256",
            image_size_y="256",
            fov="30",
            sensor_tick="0.1"
        )
        # dataset_specs : save upon creation of dataset for reproduction
        self.dataset_specs = util.AttrDict(
            map_names=self.MAP_NAMES,
            delta=self.delta,
            add_pedestrian=self.add_pedestrian,
            n_scenes=self.n_scenes,
            n_frames=self.n_frames,
            perturb_spawn_point=self.perturb_spawn_point,
            range_r=self.range_r,
            range_theta=self.range_theta,
            range_phi=self.range_phi,
            max_shift=self.max_shift,
            ped_halfwidth=self.ped_halfwidth,
            camera_attributes=self.camera_attributes,
            tally={name: 0 for name in self.MAP_NAMES}
        )

    def save_snapshot(self, map_name, scene_idx, image):
        self.dataset_specs.tally[map_name] += 1
        image.save_to_disk(f"{self.out_path}/{map_name}/scene{scene_idx}/frame{image.frame}.png")

    def attach_camera_to_spectator(self, scene_idx):
        os.makedirs(f"{self.out_path}/{self.carla_map.name}/scene{scene_idx}", exist_ok=True)
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in self.camera_attributes.items():
            blueprint.set_attribute(k, v)
        sensor = self.world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self.world.get_spectator()
        )
        save_snapshot= functools.partial(self.save_snapshot, self.carla_map.name, scene_idx)
        return sensor, save_snapshot, f"{self.carla_map.name}/scene{scene_idx}"

    def get_vehicle_blueprints(self):
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
        blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        return blueprints
    
    def get_pedestrian_blueprints(self):
        return self.world.get_blueprint_library().filter("walker.pedestrian.*")

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

    def __set_random_weather(self, vehicle=None, enable_vehicle_lights=False):
        """
        NOTE: for CARLA 0.9.11/12, not all cars have lights 
        """
        vls = carla.VehicleLightState
        weather = self.world.get_weather()
        altitude_angle = util.map_01_to_uv(-1, 90)(np.random.beta(1.3, 12))
        weather.sun_altitude_angle = altitude_angle
        deposits = util.map_01_to_uv(0, 60)(np.random.beta(0.4, 2.0))
        weather.precipitation_deposits = deposits
        cloudiness = util.map_01_to_uv(0, 100)(np.random.beta(0.4, 2.0))
        weather.cloudiness = cloudiness
        weather.wind_intensity = 50
        wetness = util.map_01_to_uv(0, 50)(np.random.beta(0.4, 2.0))
        weather.wetness = wetness
        self.world.set_weather(weather)
        if enable_vehicle_lights and vehicle and altitude_angle <= 0:
            light_state = vls(vls.LowBeam | vls.Interior | vls.Reverse | vls.Position)
            vehicle.set_light_state(light_state)

    def __set_random_frame(self, spectator, transform, vehicle=None,
            strafe_camera=False, xy_shift_camera=True):
        """
        TODO: the camera transform should be relative
        to the yaw of vehicle transform, not just vehicle location
        """
        r = random.uniform(*self.range_r)
        theta = random.uniform(*self.range_theta)
        phi = random.uniform(*self.range_phi)
        _transform = carla.Transform(
            transform.location, carla.Rotation(yaw=transform.rotation.yaw)
        )
        if xy_shift_camera:
            # NOTE: the Audi A2 has dimensions (3.70 m, 1.79 m)
            x_shift = random.uniform(-self.max_shift, self.max_shift)
            y_shift = random.uniform(-self.max_shift, self.max_shift)
            _transform.location += carla.Location(x=x_shift, y=y_shift)
        _transform = carlautil.spherical_to_camera_watcher_transform(
                r, theta, phi, pin=_transform)
        if strafe_camera and vehicle:
            # NOTE: GIRAFFE does not do camera strafing, it does camera shifting
            near_frac = vehicle.bounding_box.extent.x
            ratio_f = near_frac / self.range_r[0]
            strafe_amt = r*ratio_f - near_frac
            right_strafe = random.uniform(-strafe_amt, strafe_amt)
            above_strafe = random.uniform(-strafe_amt, strafe_amt)
            _transform = carlautil.strafe_transform(_transform,
                    right=right_strafe, above=above_strafe)

        spectator.set_transform(_transform)

    def run_scene(self, scene_idx, spawn_point):
        """Sample frames from a scene where the car is placed on some spawn point.
        """
        vehicle = None
        pedestrian = None
        camera = None
        viewing_center = None
        spectator = self.world.get_spectator()

        # set vehicle blue prints
        veh_blueprints = self.get_vehicle_blueprints()
        blueprint = random.choice(veh_blueprints)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)

        try:
            # create a vehicle
            has_vehicle = False
            spawn_point = self.__get_random_spawn_point(scene_idx)
            for _ in range(20):
                _spawn_point = carlautil.carlacopy(spawn_point)
                _spawn_point = self.__perturb_spawn_point(_spawn_point)
                vehicle = self.world.try_spawn_actor(blueprint, _spawn_point)
                if vehicle:
                    has_vehicle = True
                    spawn_point = _spawn_point
                    break

            if not has_vehicle:
                logging.warning(
                    "Failed to spawn vehicle with perturb at "
                    f"{carlautil.transform_to_str(spawn_point)}; "
                    "spawning vehicle without perturb."
                )
                vehicle = self.world.spawn_actor(blueprint, spawn_point)

            if self.show_demos:
                self.world.wait_for_tick()
            else:
                self.world.tick()
            vehicle.apply_control(carla.VehicleControl(hand_brake=True))

            if self.add_pedestrian:
                # reset the center of the scene
                off_x = random.uniform(-self.max_shift, self.max_shift)
                off_y = random.uniform(-self.max_shift, self.max_shift)
                viewing_center = carla.Transform(
                    location=spawn_point.location + carla.Vector3D(x=off_x, y=off_y),
                    rotation=spawn_point.rotation
                )

                if self.debug:
                    self.__set_random_frame(spectator, viewing_center)

                ped_blueprints = self.get_pedestrian_blueprints()
                extent = vehicle.bounding_box.extent
                r = vehicle.get_transform().rotation
                f = carlautil.location_to_ndarray(r.get_forward_vector()) * extent.x
                r = carlautil.location_to_ndarray(r.get_right_vector()) * extent.y
                l = carlautil.location_to_ndarray(vehicle.get_location())
                vertices = np.array((
                    l + f + r,
                    l - f + r,
                    l - f - r,
                    l + f - r
                ))
                vertices = util.npu.order_polytope_vertices(vertices[:, :2])
                forbidden = shapely.geometry.Polygon(vertices)

                has_pedestrian = False
                spl = np.array([viewing_center.location.x, viewing_center.location.y])
                box = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * self.ped_halfwidth
                _box = None
                for _ in range(20):
                    for _ in range(50):
                        off_x = random.uniform(-self.max_shift, self.max_shift)
                        off_y = random.uniform(-self.max_shift, self.max_shift)
                        _spl = spl + np.array([off_x, off_y])
                        _box = box + _spl
                        _poly = shapely.geometry.Polygon(_box)
                        intersection = forbidden.intersection(_poly)
                        if intersection.is_empty:
                            z = viewing_center.location.z + 0.8

                            yaw = math.degrees(random.uniform(0, 2*math.pi))
                            _spawn_point = carla.Transform(
                                location=carla.Location(x=_spl[0], y=_spl[1], z=z),
                                rotation=carla.Rotation(yaw=yaw)
                            )
                            blueprint = random.choice(ped_blueprints)
                            if self.debug:
                                carlautil.debug_polytope_on_xy_plane(
                                    self.world, vertices, z, life_time=5.
                                )
                                carlautil.debug_polytope_on_xy_plane(
                                    self.world, _box, z, life_time=5.
                                )
                            pedestrian = self.world.try_spawn_actor(blueprint, _spawn_point)
                            break

                    if pedestrian:
                        has_pedestrian = True
                        break
                
                if not has_pedestrian:
                    logging.warning(
                        "Failed to spawn pedestrian at "
                        f"{carlautil.transform_to_str(viewing_center)}; "
                        "skipping pedestrian placement."
                    )
                
            else:
                viewing_center = spawn_point

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
                self.__set_random_weather()
                # wait for car jiggle to stop
                for _ in range(20): self.world.tick()
                self.__set_random_frame(spectator, viewing_center,
                        xy_shift_camera=not self.add_pedestrian)
                self.world.tick()
                camera, save_snapshot, scene_id = self.attach_camera_to_spectator(scene_idx)
                camera.listen(save_snapshot)
                self.world.tick()
                for jdx in range(self.n_frames):
                    self.__set_random_frame(spectator, viewing_center,
                            xy_shift_camera=not self.add_pedestrian)
                    self.world.tick()
                camera.stop()
                self.world.tick()

        finally:
            if vehicle:
                vehicle.destroy()
            if pedestrian:
                pedestrian.destroy()
            if camera:
                camera.destroy()
            if self.show_demos:
                self.world.set_weather(self.original_weather)
                self.world.wait_for_tick()
            else:
                self.world.tick()
            time.sleep(2) # wait to avoid crashing

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
        time.sleep(2) # wait to avoid crashing

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
        os.makedirs(self.out_path, exist_ok=True)
        with open(os.path.join(self.out_path, "config.json"), 'w') as f:
            json.dump(self.dataset_specs, f)
