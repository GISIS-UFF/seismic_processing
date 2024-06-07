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

    sx = data.attributes(73)[traces] / data.attributes(71)[traces]
    sy = data.attributes(77)[traces] / data.attributes(71)[traces]    

    rx = data.attributes(81)[traces] / data.attributes(69)[traces]
    ry = data.attributes(85)[traces] / data.attributes(69)[traces]    

    distance = np.sqrt((sx - rx)**2 + (sy - ry)**2)

    print(distance)

    scale = 0.99*np.std(seismic)

    fig, ax = plt.subplots(num = f"Common {label} gather", ncols = 1, nrows = 1, figsize = (10, 5))

    ax.imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    x_ticks = np.arange(1, seismic.shape[1], step = 5)
    y_ticks = np.arange(0, seismic.shape[0], step = 100)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    plt.xlabel('Trace')
    plt.ylabel('Time')
    # define colorbar correctly  

    fig.tight_layout()
    plt.show()

# Davi
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
   
    plot_data = {
        "cmp": (cmpx, cmpy, 'ob'),
        "receiver": (rx, ry, 'oy'),
        "shot": (sx, sy, 'og')
    }

    plot_title = {
        "src": f"Common Shot Gatter number {index}",
        "cmp": f"Common Mid Point Gatter number {index}",
        "off": f"Common Offset Gather number {index}"
    }

    plot_order = {
        "src": ["cmp", "receiver", "shot"],
        "cmp": ["shot", "receiver", "cmp"],
        "off": ["receiver", "shot", "cmp"]
    }

    # subplots
    # 1 geometria completa
    # 2 quantidade de cmps por traço
    # 3 geometria do gather ativo

    if key in plot_order:
        for element in plot_order[key]:
            ax.plot(*plot_data[element], label=element)
    
    # to show depth with colors
    ax.scatter(rx, ry, c = rz, cmap = "viridis")
    ax.scatter(sx, sy, c = sz, cmap = "viridis")
    ax.set_title(plot_title[key], fontsize=15)
    ax.set_ylabel("Depth[m]", fontsize=10)
        
    fig.tight_layout()
    plt.gca().invert_yaxis()
    ax.legend(loc="lower left")
    plt.show()

    # define axis values according with key [almost check]
    # define labels according with key [check]
    # define colorbar correctly [pendent]

def fourier_fx_domain(data : sgy.SegyFile, key : str, index : int, fmin : float, fmax = float) -> None:
    '''
    Documentation
    
    
    '''    
    
    __check_keyword(key)

    byte, label = __keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]
   
    sx = data.attributes(73)[traces] / data.attributes(71)[traces]
   
    sy = data.attributes(77)[traces] / data.attributes(71)[traces]    

    rx = data.attributes(81)[traces] / data.attributes(69)[traces]
   
    ry = data.attributes(85)[traces] / data.attributes(69)[traces]    

    nx = len(traces)
    # distance={'cmp':np.sqrt((sx - rx)**2 + (sy - ry)**2),
    #           'off':np.sqrt((sx - rx)**2 + (sy - ry)**2),
    #           'src':np.sqrt((sx - rx)**2 + (sy - ry)**2),
    #           'rec':np.sqrt((sx - rx)**2 + (sy - ry)**2)}
    # distance=distance[key]
    # print(distance)
    # if key=='off':
    #     dx=distance[1]
    


         
    
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
    
    xloc=np.linspace(0, nx-1, 5, dtype = int)
    xlab=np.around(xloc*dx , decimals = 1)
    
    
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
    ax[1].set_xticks(xloc)
    ax[1].set_xticklabels(xlab)
    ax[1].set_title(f"Input common {label} gather")
    ax[1].set_xlabel("Offset[m]")
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
    
    floc = np.linspace(frequency[0], frequency[-1], 11, dtype = int)

    kloc = np.linspace(wavenumber[-1], wavenumber[0], 11, dtype = int)
    
    fig, ax = plt.subplots(num = f"Common {label} gather with its 1D fourier transform", ncols = 2, nrows = 1, figsize = (10, 5))

    im = ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax[0].set_yticks(tloc)
    ax[0].set_yticklabels(tlab)
    ax[0].set_xticks(xloc)
    ax[0].set_xticklabels(xlab)

    ax[0].set_title(f"Input common {label} gather")
    ax[0].set_ylabel("Two way time [s]")
    # define axis values according with key
    # define labels according with key
    # define colorbar correctly

    fk_plot = ax[1].imshow(np.abs(fk_seismic), aspect = "auto", extent=[wavenumber[0],wavenumber[-1], frequency[0], frequency[-1]], cmap = "jet")
    ax[1].set_title(f"Input FK domain")
    ax[1].set_xlabel(r"Wavenumber [m$^{-1}$]")
    ax[1].set_ylabel("Frequency [Hz]")
    
    ax[1].set_yticks(floc)
    
    fig.colorbar(im, ax=ax[0])
    fig.colorbar(fk_plot, ax=ax[1])
    fig.tight_layout()
    plt.show()

def difference(input : sgy.SegyFile, output : sgy.SegyFile, key : str, index : int) -> None:
    '''
    Documentation
    
    
    '''    
    __check_keyword(key)

    byte, label = __keywords.get(key)

    traces = np.where(input.attributes(byte)[:] == index)[0]

    seismic_input = input.trace.raw[:].T
    seismic_input = seismic_input[:, traces]

    seismic_output = output.trace.raw[:].T
    seismic_output = seismic_output[:, traces]

    seismic_diff = seismic_output - seismic_input

    scale = 0.99*np.std(seismic_input)

    fig, ax = plt.subplots(num = f"Common {label} gather", ncols = 3, nrows = 1, figsize = (18, 5))

    ax[0].imshow(seismic_input, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)


    ax[1].imshow(seismic_output, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)


    ax[2].imshow(seismic_diff, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    # define axis values according with key
    # define labels according with key
    # define colorbar correctly
        
    fig.tight_layout()
    plt.show()

