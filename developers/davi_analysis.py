from sys import path
import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
path.append("../")

from toolbox import managing as mng
from toolbox import visualizing as view

PATH = "/home/malum/seismic_processing/data/overthrust_synthetic_seismic_data" 

data = mng.import_sgy_file(PATH)

def fourier_fk_domain(data: sgy.SegyFile, **kwargs) -> sgy.SegyFile:

    current_clicks = []
    polygons = []
    polygon_paths = []
    masked_image = None
    ready = False  # Novo controle de fluxo

    def on_click(event):
        nonlocal current_clicks
        if event.inaxes == ax[1]:
            current_clicks.append((event.xdata, event.ydata))
            ax[1].plot(event.xdata, event.ydata, 'ro')
            plt.draw()

            if len(current_clicks) >= 3:
                for artist in ax[1].patches + ax[1].lines:
                    artist.remove()

                for polygon in polygons:
                    ax[1].add_patch(polygon)

                polygon = Polygon(current_clicks, closed=True, edgecolor='black', facecolor='cyan', alpha=0.3)
                ax[1].add_patch(polygon)
                plt.draw()

    def on_key(event):
        nonlocal current_clicks, polygons, polygon_paths, masked_image, ready  # Adicionar ready
        if event.key == 'n':
            if len(current_clicks) >= 3:
                polygon = Polygon(current_clicks, closed=True, edgecolor='black', facecolor='cyan', alpha=0.3)
                polygons.append(polygon)
                polygon_path = Path(current_clicks)
                polygon_paths.append(polygon_path)

                teste = np.abs(fk_seismic[mask,:][::-1])
                gaussian_mask = np.zeros(teste.shape)

                x, y = np.meshgrid(np.arange(teste.shape[1]), np.arange(teste.shape[0]))
                xy = np.vstack((x.flatten(), y.flatten())).T

                for polygon_path in polygon_paths:
                    inside = polygon_path.contains_points(xy).reshape(gaussian_mask.shape)
                    gaussian_mask[inside] = 1

                gaussian_mask = gaussian_filter(gaussian_mask.astype(np.float32), sigma=7)

                masked_image = np.abs(teste) * gaussian_mask
                ready = True  # Marca o estado como pronto

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(masked_image, aspect="auto", cmap="jet")
                ax.set_title('Filtered F-K', fontsize=15)

                def on_close(event):
                    polygons.clear()
                    polygon_paths.clear()
                    for artist in ax[1].patches + ax[1].lines:
                        artist.remove()
                    plt.draw()

                fig.canvas.mpl_connect('close_event', on_close)

                plt.show()

            current_clicks = []

    fmax = kwargs.get("fmax") if "fmax" in kwargs else 100.0

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.get_keyword_indexes(data, key)[0]

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte = mng.__keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]
        
    distance = data.attributes(37)[traces] / data.attributes(69)[traces]

    nx = len(traces)
    dh = np.median(np.abs(np.abs(distance[1:]) - np.abs(distance[:-1]))) 

    fk_seismic = np.fft.fftshift(np.fft.fft2(seismic))

    frequency = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    wavenumber = np.fft.fftshift(np.fft.fftfreq(nx, dh))

    mask = np.logical_and(frequency >= 0.0, frequency <= fmax)

    xloc = np.linspace(0, len(traces)-1, 5, dtype=int)
    xlab = traces[xloc]

    tloc = np.linspace(0, nt-1, 11, dtype=int)
    tlab = np.around(tloc*dt, decimals=1)

    floc = np.linspace(0, len(frequency[mask])-1, 11, dtype=int)
    flab = np.around(np.ceil(frequency[mask][floc][::-1]), decimals=1)

    kloc = np.linspace(0, len(wavenumber)-1, 5, dtype=int)
    klab = np.around(wavenumber[kloc], decimals=3)

    scale = 0.8 * np.std(seismic)
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    im = ax[0].imshow(seismic, aspect="auto", cmap="Greys", vmin=-scale, vmax=scale)

    ax[0].set_yticks(tloc)
    ax[0].set_yticklabels(tlab)
    ax[0].set_xticks(xloc)
    ax[0].set_xticklabels(xlab)

    ax[0].set_ylabel('Time [s]', fontsize=15)
    ax[0].set_xlabel('Trace number', fontsize=15)

    fk = ax[1].imshow(np.abs(fk_seismic[mask, :][::-1]), aspect="auto", cmap="jet")
    
    ax[1].set_yticks(floc)
    ax[1].set_yticklabels(flab)

    ax[1].set_xticks(kloc)
    ax[1].set_xticklabels(klab)

    ax[1].set_ylabel("Frequency [Hz]", fontsize=15)
    ax[1].set_xlabel(r"Wavenumber [m$^{-1}$]", fontsize=15)

    ax[0].cbar = fig.colorbar(im, ax=ax[0])
    ax[0].cbar.set_label("Amplitude", fontsize=15) 

    ax[1].cbar = fig.colorbar(fk, ax=ax[1])
    ax[1].cbar.set_label("Amplitude", fontsize=15) 
   
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    fig.tight_layout()
    plt.show()

    return masked_image

def fold_geometry(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot a fold map, cmp coodinates the current configuration according 
    to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    ### Examples:

    >>> view.fold_geometry(data)                           # plots first shot 
    >>> view.fold_geometry(data, key = "off")              # plots first offset
    >>> view.fold_geometry(data, key = "rec", index = 35)  # plots rec index 789
    >>> view.fold_geometry(data, key = "cmp", index = 512) # plots cmp index 512
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (15, 5))

    cmp = data.attributes(21)[:] / data.attributes(69)[:]
    cmpx,cmpt=np.unique(cmp,return_counts=True)

    im = ax.scatter(cmpx, cmpt, c=cmpt, cmap='jet', label="CMPs")
    ax.set_xlabel("Distance [km]", fontsize=15)
    ax.set_ylabel("CDP Fold Geometry", fontsize=15)

    cax = fig.colorbar(im, ax=ax)
    cax.set_ticks(np.linspace(cmpt.min(), cmpt.max(), num=5).astype(int))
    cax.set_label("CDP Fold Geometry", fontsize=15)

    fig.tight_layout()
    plt.show()

mng.show_trace_header(data)

masked_image = fourier_fk_domain(data)

# fold_geometry(data, key = "rec", index = 35)

