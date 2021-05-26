"""Run the CARLA server and take a picture
"""
import os
import time
import subprocess
import signal
import atexit
import carla

CARLA_HOST = '127.0.0.1'
CARLA_PORT = 3000
CARLA_MAP = 'Town03'

binpath = os.path.join(os.environ['CARLA_DIR'], 'CarlaUE4.sh')
# "DISPLAY= ./CarlaUE4.sh -opengl"
settings = f"-quality-level=Low -carla-rpc-port={CARLA_PORT}"
cmd = ' '.join([binpath, settings]).split()
print("Calling", *cmd)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
# atexit.register(os.killpg, proc.pid, signal.SIGKILL)
time.sleep(5.0)
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(10.0)
print("Made client.")
print("Housekeeping...")
world = client.get_world()
carla_map = world.get_map()
if carla_map.name != CARLA_MAP:
    world = client.load_world(CARLA_MAP)
    carla_map = world.get_map()
traffic_manager = client.get_trafficmanager(8000)

# Take a picture
print("Preparing sensor.")

# Find the blueprint of the sensor.
blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
blueprint.set_attribute('image_size_x', '500')
blueprint.set_attribute('image_size_y', '500')
blueprint.set_attribute('fov', '90')
blueprint.set_attribute('sensor_tick', '0.1')
spawn_point = carla_map.get_spawn_points()[0]
sensor = world.spawn_actor(blueprint, spawn_point)
has_picture = False

def take_picture(data):
    print("Took picture.")
    data.save_to_disk('out/my_picture.png')
    global sensor
    global proc
    global has_picture
    sensor.stop()
    sensor.destroy()
    has_picture = True
    # Kill CARLA server using process group ID
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    print("Shutting down.")
    
sensor.listen(lambda data: take_picture(data))
print("Sensor is listening.")

while not has_picture:
    time.sleep(1)
print("Shut down.")
