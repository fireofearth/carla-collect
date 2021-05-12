
import carla
import argparse
import random
import time

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
argparser.add_argument(
    '-d', '--delta',
    metavar='D',
    default=0.05,
    type=float,
    help='Simulation synchronous mode timestep (default: 0.05)')
argparser.add_argument(
    '--filter',
    metavar='PATTERN',
    default='model3',
    help='Actor (e.g. car model) to use as ego vehicle (default: "vehicle.*")')
args = argparser.parse_args()

"""Create and connect CARLA client
https://carla.readthedocs.io/en/latest/python_api/#carlaclient
"""
client = carla.Client(args.host, args.port)

"""Change the map
https://carla.readthedocs.io/en/latest/core_map/#changing-the-map
Not necessary if wishing to use the default Town03
"""
# world = client.load_world('Town03')
world = client.get_world()

"""Get settings object for this world
https://carla.readthedocs.io/en/latest/python_api/#carla.WorldSettings

synchronous_mode (bool)
When set to true, the server will wait for a client tick in order to move forward.
It is false by default.
"""
original_settings = world.get_settings()
settings = world.get_settings()
# settings.synchronous_mode = True
# settings.fixed_delta_seconds = args.delta
# settings.no_rendering_mode = arg.no_rendering
world.apply_settings(settings)

# traffic_manager = client.get_trafficmanager(8000)
# traffic_manager.set_synchronous_mode(True)

"""
Create an ego vehicle from blue print
https://carla.readthedocs.io/en/latest/core_actors/
"""
blueprint_library = world.get_blueprint_library()
vehicle_blueprint = blueprint_library.filter(args.filter)[0]
assert(int(vehicle_blueprint.get_attribute('number_of_wheels')) == 4)
vehicle_transform = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_blueprint, vehicle_transform)

"""
/carla/PythonAPI/carla/agents/navigation
contains configurations for different autopilot agents.
"""
vehicle.set_autopilot(True)

# lidar_bp = generate_lidar_bp(arg, world, blueprint_library, args.delta)

# user_offset = carla.Location(arg.x, arg.y, arg.z)
# lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)

# lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# point_list = o3d.geometry.PointCloud()
# if arg.semantic:
#     lidar.listen(lambda data: semantic_lidar_callback(data, point_list))
# else:
#     lidar.listen(lambda data: lidar_callback(data, point_list))

# lists the possible maps to load
# print(client.get_available_maps())

time.sleep(10)

vehicle.destroy()
world.apply_settings(original_settings)
