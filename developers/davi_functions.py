import numpy as np
import matplotlib.pyplot as plt
from sys import path

path.append("../")
from toolbox import managing as mng

def load_seismic(data, key, index):

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte = mng.__keywords.get(key)
    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    traces = np.where(data.attributes(byte)[:] == index)[0]

    seismic = np.zeros((nt, len(traces)))

    for i in range(len(traces)):
        seismic[:,i] = data.trace.raw[traces[i]] 
        
    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)

    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)

    return seismic, [xloc, tloc, tlab]

def plot_seismic(seismic, tickNlabels):
    xloc, tloc, tlab = tickNlabels

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (10, 5))

    scale = 0.9*np.std(seismic)
    im = ax.imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax.set_xticks(xloc)
    ax.set_yticks(tloc)
    ax.set_xticklabels(xloc)
    ax.set_yticklabels(tlab)
    ax.set_ylabel('Time [s]', fontsize = 15)
    ax.set_xlabel('Relative trace number', fontsize = 15)
    ax.cbar = fig.colorbar(im)
    ax.cbar.set_label("Amplitude", fontsize = 15) 

    plt.show()

