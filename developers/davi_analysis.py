from sys import path
import scipy as sc
import numpy as np
import segyio as sgy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
path.append("../")

from toolbox import managing as mng
from toolbox import visualizing as view

data_file = "../data/2D_Land_vibro_data_2ms/Line_001.sgy"

data = mng.import_sgy_file(data_file)

current_clicks = []
polygons = []
polygon_paths = []

def on_click(event):
    if event.inaxes:
        current_clicks.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'ro')
        plt.draw()

        if len(current_clicks) >= 3:
            for artist in ax.patches:
                artist.remove()

            for polygon in polygons:
                ax.add_patch(polygon)

            polygon = Polygon(current_clicks, closed=True, edgecolor='black', facecolor='cyan', alpha=0.5)
            ax.add_patch(polygon)
            plt.draw()

def on_key(event):
    if event.key == 'n':
        if len(current_clicks) >= 3:
            polygon = Polygon(current_clicks, closed=True, edgecolor='black', facecolor='cyan', alpha=0.5)
            polygons.append(polygon)
            polygon_path = Path(current_clicks)
            polygon_paths.append(polygon_path)

        current_clicks = []

fig, ax = plt.subplots()

ax.imshow(data)

cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()

id = 0 # 0 - filter false; 1 - filter true

if polygon_paths:
    mask = np.ones(data.shape[:2])

    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    xy = np.vstack((x.flatten(), y.flatten())).T

    for polygon_path in polygon_paths:
        inside = polygon_path.contains_points(xy).reshape(mask.shape)
        mask[inside] = id 

    mask = gaussian_filter(mask.astype(np.float32), sigma=10)

    mask = mask / np.max(mask)

    masked_image = data * mask

    plt.figure()
    plt.imshow(masked_image.astype(np.uint16))
    plt.title("Image with Mask Applied")
    plt.show()


