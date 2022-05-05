import collections

from collect.generate.map import CachedMapData

def test_cached_map_data():
    cached_map_data = CachedMapData()
    map_name = "Town03"
    map_data = cached_map_data.map_datum[map_name]
    # map data contains these attributes
    keys = ['road_polygons', 'white_lines', 'yellow_lines', 'junctions']
    for key in keys: assert key in map_data
    # there are 31 junctions in Town03
    assert len(map_data['junctions']) == 31

    # a junction contains these attributes
    keys = ['pos', 'waypoints']
    for key in keys: assert key in map_data['junctions'][0]
    # pos is an (x,y) position
    assert map_data['junctions'][0]['pos'].shape == (2,)
    
    shape = map_data['junctions'][0]['waypoints'].shape
    # these are pairs of wapoints entering and exiting the junction
    assert shape[1] == 2
    # each waypoint is vector of (x position, y position, yaw, road length)
    assert shape[2] == 4
    