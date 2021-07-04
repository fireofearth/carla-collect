
import carla

# Read the .osm data
with open("maps/map.osm", 'r') as f:
    osm_data = f.read()

settings = carla.Osm2OdrSettings()
xodr_data = carla.Osm2Odr.convert(osm_data, settings)

# save opendrive file
with open("maps.xodr", 'w') as f:
    f.write(xodr_data)
