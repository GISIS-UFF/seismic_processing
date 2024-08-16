from sys import path
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

view.fourier_fk_domain(data)

def fourier_fk_domain(data: sgy.SegyFile, angle: float) -> sgy.SegyFile:
    current_clicks = []
    polygons = []
    polygon_paths = []

    def on_click(event):
        global current_clicks
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
        global current_clicks, polygons, polygon_paths
        if event.key == 'n':
            if len(current_clicks) >= 3:
                polygon = Polygon(current_clicks, closed=True, edgecolor='black', facecolor='cyan', alpha=0.5)
                polygons.append(polygon)
                polygon_path = Path(current_clicks)
                polygon_paths.append(polygon_path)

            current_clicks = []

    fig, ax = plt.subplots()
    ax.imshow(data, cmap='gray')
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if polygon_paths:
        mask = np.zeros(data.shape)

        x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        xy = np.vstack((x.flatten(), y.flatten())).T

        for polygon_path in polygon_paths:
            inside = polygon_path.contains_points(xy).reshape(mask.shape)
            mask[inside] = 1

        mask = gaussian_filter(mask.astype(np.float32), sigma=7)

        masked_image = data * mask

        plt.figure()
        plt.imshow(masked_image, cmap='gray')
        plt.show()


