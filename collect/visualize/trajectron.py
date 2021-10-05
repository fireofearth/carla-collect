import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.animation as animation
import matplotlib.patches as patches
import cv2 as cv


def render_entire_map(ax, map_data):
    for poly in map_data.road_polygons:
        patch = patches.Polygon(poly[:, :2], fill=True, color='grey')
        ax.add_patch(patch)
    for line in map_data.yellow_lines:
        ax.plot(line.T[0], line.T[1], c='yellow', linewidth=2)
    for line in map_data.white_lines:
        ax.plot(line.T[0], line.T[1], c='white', linewidth=2)


def render_map_crop(ax, map_data, extent, pixels_per_m=3):
    """Render crop of a map from road polygons defined by extent
    
    Parameters
    ==========
    ax : matplotlib.pyplot.Axes
        The axes to render onto.
    map_data : util.AttrDict
        The lines and polygon ndarrays of the lanes and roads representing the map. 
    extent : tuple of int
        The extent of the map to render of form (x_min, x_max, y_min, y_max) in meters
    pixels_per_m : int
        The number of pixels per meter.
    """
    x_min, x_max, y_min, y_max = extent
    x_size = x_max - x_min
    y_size = y_max - y_min
    dim = (int(pixels_per_m * y_size), int(pixels_per_m * x_size), 3)
    image = np.zeros(dim)

    for polygon in map_data.road_polygons:
        rzpoly = ( pixels_per_m*(polygon[:,:2] - np.array([x_min, y_min])) ).astype(int).reshape((-1,1,2))
        cv.fillPoly(image, [rzpoly], (150,150,150))

    for line in map_data.white_lines:
        rzline = ( pixels_per_m*(line[:,:2] - np.array([x_min, y_min])) ).astype(int).reshape((-1,1,2))
        cv.polylines(image, [rzline], False, (255,255,255), thickness=2)

    for line in map_data.yellow_lines:
        rzline = ( pixels_per_m*(line[:,:2] - np.array([x_min, y_min])) ).astype(int).reshape((-1,1,2))
        cv.polylines(image, [rzline], False, (255,255,0), thickness=2)

    image = image.astype(np.uint8).swapaxes(0,1)
    ax.imshow(image.swapaxes(0,1), extent=extent, origin='lower', interpolation='none')


def render_scene(ax, scene, white_road=False, global_coordinates=False):
    """Render road overlay of a scene.

    Parameters
    ==========
    ax : matplotlib.axes.Axes
        axes to render road into.
    scene : Scene
        Scene to render road overlay.
    white_road : bool
        Whether to fill roads grey (False) or white (True), grey by default.
    global_coordinates : bool
        Whether to use local coordinates (0, x-size, 0, y-size) (False) or
        global coordinates (x-min, x-max, y-min, y-max) (True). Local by default.
    """
    road_color = 'white' if white_road else 'grey'
    map_mask = scene.map['VEHICLE'].as_image()
    # map_mask has shape (y, x, c)
    road_bitmap = np.max(map_mask, axis=2)
    road_div_bitmap = map_mask[..., 1]
    lane_div_bitmap = map_mask[..., 0]
    if global_coordinates:
        extent = (scene.x_min, scene.x_max, scene.y_min, scene.y_max)
    else:
        extent = (0, scene.x_size, 0, scene.y_size)
    """
    NuScenes bitmap format
    scene.map[...].as_image() has shape (y, x, c)
    Channel 1: lane, road_segment, drivable_area
    Channel 2: road_divider
    Channel 3: lane_divider
    """
    road_bitmap = np.max(map_mask, axis=2)
    road_div_bitmap = map_mask[..., 1]
    lane_div_bitmap = map_mask[..., 2]
    ax.imshow(road_bitmap,     extent=extent, origin='lower', cmap=colors.ListedColormap(['none', road_color]))
    ax.imshow(road_div_bitmap, extent=extent, origin='lower', cmap=colors.ListedColormap(['none', 'yellow']))
    ax.imshow(lane_div_bitmap, extent=extent, origin='lower', cmap=colors.ListedColormap(['none', 'silver']))

