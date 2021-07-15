import glob
import os
import sys
import random
import time
import math
from collections import deque

import cv2 as cv
import numpy as np
import carla

SHOW_PREVIEW = False
IM_WIDTH  = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = 'Xception'
MEMORY_FRACTION = 0.8
MIN_REWARD = -200

EPISODES = 100
DISCOUTN = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.carla_map = world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        self.transform = random.choice(self.carla_map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        self.cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.cam_bp.set_attribute("image_size_x", f"{self.im_width}")
        self.cam_bp.set_attribute("image_size_y", f"{self.im_height}")
        self.cam_bp.set_attribute("fov", "110")
        # self.cam_bp.set_attribute('sensor_tick', '1.0')
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.cam_bp, transform,
                attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        control = carla.VehicleControl(throttle=0.0, steer=0.0)
        self.vehicle.apply_control(control)
        time.sleep(4)
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform,
                attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)
        
        self.episode_start = time.time()
        control = carla.VehicleControl(throttle=0.0, steer=0.0)
        self.vehicle.apply_control(control)

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        print("In process_img")
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        if self.SHOW_CAM:
            pass
        i3 = i2[:, :, :3]
        self.front_camera = i3
        # return i3/255.0

    def step(self, action):
        if action == 0:
            # left
            control = carla.VehicleControl(
                    throttle=1.0, steer=-1.0*self.STEER_AMT)
        elif action == 1:
            # straight
            control = carla.VehicleControl(
                    throttle=1.0, steer=0)
        elif action == 2:
            control = carla.VehicleControl(
                    throttle=1.0, steer=1.0*self.STEER_AMT)
        self.vehicle.apply_control(control)
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        # return obs, reward, done, extra_info
        return self.front_camera, reward, done, None

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
    
    def create_model(self):
        base_model = Xception(weights=None, )

def process_img(image):
    print("In process_img")
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    return i3/255.0

actor_list = []

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter("model3")[0]

    carla_map = world.get_map()
    # spawn_point = random.choice(carla_map.get_spawn_points())
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn_point)
    actor_list.append(vehicle)

    control = carla.VehicleControl(throttle=1.0, steer=0.0)
    vehicle.apply_control(control)

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")
    cam_bp.set_attribute('sensor_tick', '1.0')

    spawn_point = carla.Transform(carla.Location(x=2.5, z=1.7))
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)

    sensor.listen(lambda data: process_img(data))
    time.sleep(10)

finally:
    print("clean up")
    for actor in actor_list:
        actor.destroy()
    # clean up
