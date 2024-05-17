import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

__keywords = {'src' : [9,  'shot'], 
              'rec' : [13, 'receiver'], 
              'off' : [37, 'offset'], 
              'cmp' : [21, 'mid point']}

def __check_keyword(key : str) -> None:
    '''
    Documentation
    

    '''    
    
    if key not in __keywords.keys():
        print("Invalid keyword!")
        print("Please use a valid header keyword: ['src', 'rec', 'off', 'cmp']")
        exit()

def keyword_indexes(data : sgy.SegyFile, key : str) -> np.ndarray:
    '''
    Documentation
    

    '''    

    __check_keyword(key)

    byte = __keywords.get(key)[0]
    possibilities = np.unique(data.attributes(byte))

    return possibilities

# Amanda
def seismic(data : sgy.SegyFile, key : str, index : int) -> None:
    '''
    Plot a seismic gather according with a specific header keyword.
    
    Parameters
    ----------        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    Examples
    --------
    >>> plot_seismic(data, key = "src", index = 51)
    >>> plot_seismic(data, key = "rec", index = 203)
    >>> plot_seismic(data, key = "cmp", index = 315)
    >>> plot_seismic(data, key = "off", index = 223750)
    '''    

    __check_keyword(key)

    byte, label = __keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    scale = 0.99*np.std(seismic)

    fig, ax = plt.subplots(num = f"Common {label} gather", ncols = 1, nrows = 1, figsize = (10, 5))

    ax.imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    # define axis values according with key
    # define labels according with key
    # define colorbar correctly
        
    fig.tight_layout()
    plt.show()

# David
def geometry(data : sgy.SegyFile, key : str, index : int) -> None:
    '''
    Documentation
    
    
    '''    

    __check_keyword(key)

    byte, label = __keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    sx = data.attributes(73)[traces] / data.attributes(71)[traces]
    sy = data.attributes(77)[traces] / data.attributes(71)[traces]    
    sz = data.attributes(45)[traces] / data.attributes(71)[traces]

    rx = data.attributes(81)[traces] / data.attributes(69)[traces]
    ry = data.attributes(85)[traces] / data.attributes(69)[traces]    
    rz = data.attributes(41)[traces] / data.attributes(69)[traces]

    cmpx = data.attributes(181)[traces] / data.attributes(69)[traces]
    cmpy = data.attributes(185)[traces] / data.attributes(69)[traces]    

    fig, ax = plt.subplots(num = f"Common {label} gather", ncols = 1, nrows = 1, figsize = (10, 5))

    # adjust order of plots according with key
    # example: 
    #    src key order -> 1 cmp, 2 receiver and 3 shot
    #    cmp key order -> 1 shot, 2 receiver and 3 cmp          

    ax.plot(cmpx, cmpy, 'ob') # 1
    ax.plot(rx, ry, 'oy')     # 2
    ax.plot(sx, sy, 'og')     # 3

    # to show depth with colors
    # ax.scatter(rx, rx, c = rz, cmap = ???)
    # ax.scatter(sx, sx, c = sz, cmap = ???)

    # define axis values according with key
    # define labels according with key
    # define colorbar correctly
        
    fig.tight_layout()
    plt.show()

# Anthony
def fourier_fx_domain(data : sgy.SegyFile, key : str, index : int, fmin : float, fmax = float) -> None:
    '''
    Documentation
    
    
    '''    
    
    __check_keyword(key)

    byte, label = __keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    nx = len(traces)
    dx = 25.0  # choose according with input key
    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    frequency = np.fft.fftfreq(nt, dt)
    fx_seismic = np.fft.fft(seismic, axis = 0)

    for i in range(len(traces)):
        fx_seismic[:,i] *= 1.0 / np.max(fx_seismic[:,i]) 

    scale = 0.99*np.std(seismic)

    mask = np.logical_and(frequency >= fmin, frequency <= fmax)

    floc = np.linspace(0, len(frequency[mask]), 11, dtype = int)
    flab = np.around(frequency[floc], decimals = 1)
    
    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)

    fig, ax = plt.subplots(num = f"Common {label} gather with its 1D fourier transform", ncols = 2, nrows = 1, figsize = (10, 5))

    ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax[0].set_yticks(tloc)
    ax[0].set_yticklabels(tlab)
    # ax[0].set_xticks(xloc)
    # ax[0].set_xticklabels(xlab)

    ax[0].set_title(f"Input common {label} gather")
    ax[0].set_ylabel("Two way time [s]")
    # define axis values according with key
    # define labels according with key
    # define colorbar correctly

    ax[1].imshow(np.abs(fx_seismic[mask,:]), aspect = "auto", cmap = "jet")
    ax[1].set_yticks(floc)
    ax[1].set_yticklabels(flab)
    # ax[1].set_xticks(xloc)
    # ax[1].set_xticklabels(xlab)
    ax[1].set_title(f"Input common {label} gather")
    ax[1].set_ylabel("Frequency [Hz]")

    fig.tight_layout()
    plt.show()

# Jonatas
def fourier_fk_domain(data : sgy.SegyFile, key : str, index : int, fmin : float, fmax = float) -> None:
    '''
    Documentation
    
    
    '''    
    
    __check_keyword(key)

    byte, label = __keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    nx = len(traces)
    dx = 25.0  # choose according with input key
    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    fk_seismic = np.fft.fftshift(np.fft.fft2(seismic))

    frequency = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    wavenumber = np.fft.fftshift(np.fft.fftfreq(nx, dx))

    scale = 0.99*np.std(seismic)

    xloc = np.linspace(0, nx, 5)
    xlab = np.around(xloc*dx, decimals = 1)

    tloc = np.linspace(0, nt, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)

    fig, ax = plt.subplots(num = f"Common {label} gather with its 1D fourier transform", ncols = 2, nrows = 1, figsize = (10, 5))

    ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax[0].set_yticks(tloc)
    ax[0].set_yticklabels(tlab)
    # ax[0].set_xticks(xloc)
    # ax[0].set_xticklabels(xlab)

    ax[0].set_title(f"Input common {label} gather")
    ax[0].set_ylabel("Two way time [s]")
    # define axis values according with key
    # define labels according with key
    # define colorbar correctly

    ax[1].imshow(np.abs(fk_seismic), aspect = "auto", cmap = "jet")
    ax[1].set_title(f"Input FK domain")
    ax[1].set_xlabel(r"Wavenumber [m$^{-1}$]")
    ax[1].set_ylabel("Frequency [Hz]")

    fig.tight_layout()
    plt.show()

def difference(data1 : sgy.SegyFile, data2 : sgy.SegyFile) -> None:
    '''
    Documentation
    
    
    '''    

    pass    

