import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

__keywords = {'src' : [9,  'shot'], 
              'rec' : [13, 'receiver'], 
              'off' : [37, 'offset'], 
              'cmp' : [21, 'mid point']}

def __check_keyword(key : str) -> None:
    
    if key not in __keywords.keys():
        print("\033[31mInvalid keyword!\033[m\
                     \nPlease use a valid header keyword: ['src', 'rec', 'off', 'cmp']")
        exit()

def __check_index(data : sgy.SegyFile, key : str, index : int ) -> None:   
    
    if index not in keyword_indexes(data, key):
        print("\033[31mInvalid index choice!\033[m\
                     \nPlease use the function \033[33mkeyword_indexes\033[m to choose a properly index.")
        exit()

def keyword_indexes(data : sgy.SegyFile, key : str) -> np.ndarray:
    '''
    Print possible indexes to access in seismic gather.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    ### Examples:
    
    >>> keyword_indexes(data, key = "src")
    >>> keyword_indexes(data, key = "rec")
    >>> keyword_indexes(data, key = "cmp")
    >>> keyword_indexes(data, key = "off")
    '''    

    __check_keyword(key)

    byte = __keywords.get(key)[0]

    return np.unique(data.attributes(byte))

def seismic(data : sgy.SegyFile, key : str, index : int) -> None:
    '''
    Plot a seismic gather according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    ### Examples:

    >>> view.seismic(data, key = "src", index = 51)
    >>> view.seismic(data, key = "rec", index = 203)
    >>> view.seismic(data, key = "cmp", index = 315)
    >>> view.seismic(data, key = "off", index = 223750)
    '''    

    __check_keyword(key)
    __check_index(data, key, index)

    byte, label = __keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    scale = 0.9*np.std(seismic)

    fig, ax = plt.subplots(num = f"Common {label} gather", ncols = 1, nrows = 1, figsize = (10, 5))

    img = ax.imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]

    tloc = np.linspace(0, nt, 11, dtype = int)
    tlab = np.around(tloc * dt, decimals = 1)
    
    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)
    
    ax.set_yticks(tloc)
    ax.set_yticklabels(tlab)

    ax.set_ylabel('Time [s]', fontsize = 15)
    ax.set_xlabel('Trace number', fontsize = 15)

    cbar = fig.colorbar(img, ax = ax)
    cbar.set_label("Amplitude", fontsize = 15)

    fig.tight_layout()
    plt.show()

def geometry(data : sgy.SegyFile, key : str, index : int) -> None:
    '''
    Plot geometry, traces per cmp and the current source-receiver configuration 
    according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    ### Examples:

    >>> view.geometry(data, key = "src", index = 51)
    >>> view.geometry(data, key = "rec", index = 203)
    >>> view.geometry(data, key = "cmp", index = 315)
    >>> view.geometry(data, key = "off", index = 223750)
    '''    

    __check_keyword(key)
    __check_index(data, key, index)

    byte, label = __keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    sx_test = data.attributes(73)[:] / data.attributes(71)[:]
    sz_test = data.attributes(73)[:] / data.attributes(71)[:]


    sx = data.attributes(73)[traces] / data.attributes(71)[traces]
    sy = data.attributes(77)[traces] / data.attributes(71)[traces]    
    sz = data.attributes(45)[traces] / data.attributes(71)[traces]

    rx = data.attributes(81)[traces] / data.attributes(69)[traces]
    ry = data.attributes(85)[traces] / data.attributes(69)[traces]    
    rz = data.attributes(41)[traces] / data.attributes(69)[traces]

    cmpx = data.attributes(181)[traces] / data.attributes(69)[traces]
    cmpy = data.attributes(185)[traces] / data.attributes(69)[traces]

    cmp_trace, traces_per_cmp = np.unique(data.attributes(25)[:], return_counts = True)          
   
    print(cmp_trace, traces_per_cmp)

    plot_data = {
        "cmp": (cmpx, cmpy, 'ob'),
        "receiver": (rx, ry, 'oy'),
        "shot": (sx, sy, 'og')
    }

    plot_title = {
        "src": f"Common Shot Gather number {index}",
        "cmp": f"Common Mid Point Gather number {index}",
        "off": f"Common Offset Gather number {index}"
    }

    plot_order = {
        "src": ["cmp", "receiver", "shot"],
        "cmp": ["shot", "receiver", "cmp"],
        "off": ["receiver", "shot", "cmp"]
    }
    
    fig, ax = plt.subplots(num = f"Common {label} gather", nrows = 3, ncols = 1, figsize = (10, 5))

    ax[2].scatter(sx, sy, c = sz, cmap = "viridis", label="Sources")
    ax[2].set_title("Geometry", fontsize=15)
    im2 = ax[2].scatter(rx, ry, c = rz, cmap = "viridis", label="Receivers")
    ax[2].cbar = fig.colorbar(im2, ax = ax[2])
    ax[2].cbar.set_label("Receiver Depth", fontsize = 15) # não consegui entender como funciona esse "c = arg"
        
    ax[1].scatter(cmpx, cmpy, label="CMP per Trace")


    if key in plot_order:
        for element in plot_order[key]:
            ax[0].plot(*plot_data[element], label=element)
            ax[0].set_title(plot_title[key], fontsize=15)

    for i in range(len(ax)):
        ax[i].set_xlabel("Distance [m]", fontsize=12)
        ax[i].legend(loc="lower left")
        ax[i].set_xticks(np.linspace(rx.min(), rx.max(), 11, dtype=int))
        ax[i].set_xticklabels(np.array(np.linspace(0, 15000, 11, dtype=int))) # Como que eu consigo achar esse 15k de forma legit(?)

    fig.tight_layout()
    plt.gca().invert_yaxis()

    plt.show()

def fourier_fx_domain(data : sgy.SegyFile, key : str, index : int, fmin : float, fmax = float) -> None:
    '''
    Plot the amplitude spectra of each trace in gather according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    fmin: minimum frequency to visualize    
    
    fmin: maximum frequency to visualize    
    
    ### Examples:

    >>> view.fourier_fx_domain(data, key = "src", index = 51, fmin = 0, fmax = 100)
    >>> view.fourier_fx_domain(data, key = "rec", index = 203, fmin = 0, fmax = 100)
    >>> view.fourier_fx_domain(data, key = "cmp", index = 315, fmin = 0, fmax = 100)
    >>> view.fourier_fx_domain(data, key = "off", index = 223750, fmin = 0, fmax = 100)
    '''    
    
    __check_keyword(key)
    __check_index(data, key, index)

    byte, label = __keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]    
    
    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6
    
    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    frequency = np.fft.fftfreq(nt, dt)
    fx_seismic = np.fft.fft(seismic, axis = 0)

    for i in range(len(traces)):
        fx_seismic[:,i] *= 1.0 / np.max(fx_seismic[:,i]) 

    scale = 0.9*np.std(seismic)

    mask = np.logical_and(frequency >= fmin, frequency <= fmax)

    floc = np.linspace(0, len(frequency[mask])-1, 11, dtype = int)
    flab = np.around(frequency[floc], decimals = 1)
    
    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]
    
    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)

    fig, ax = plt.subplots(num = f"Common {label} gather with its 1D fourier transform", ncols = 2, nrows = 1, figsize = (10, 5))

    im = ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax[0].set_yticks(tloc)
    ax[0].set_yticklabels(tlab)
    ax[0].set_xticks(xloc)
    ax[0].set_xticklabels(xlab)

    ax[0].set_ylabel('Time [s]', fontsize = 15)
    ax[0].set_xlabel('Trace number', fontsize = 15)
    
    fx = ax[1].imshow(np.abs(fx_seismic[mask,:]), aspect = "auto", cmap = "jet")
   
    ax[1].set_yticks(floc)
    ax[1].set_yticklabels(flab)
    ax[1].set_xticks(xloc)
    ax[1].set_xticklabels(xlab)

    ax[1].set_ylabel("Frequency [Hz]", fontsize = 15)
    ax[1].set_xlabel("Trace number", fontsize = 15)

    ax[0].cbar = fig.colorbar(im, ax = ax[0])
    ax[0].cbar.set_label("Amplitude", fontsize = 15) 

    ax[1].cbar = fig.colorbar(fx, ax = ax[1])
    ax[1].cbar.set_label("Amplitude", fontsize = 15) 

    fig.tight_layout()
    plt.show()

def fourier_fk_domain(data : sgy.SegyFile, key : str, index : int, fmin : float, fmax = float) -> None:
    '''
    Plot the amplitude spectra of each trace in gather according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    fmin: minimum frequency to visualize    
    
    fmin: maximum frequency to visualize    
    
    ### Examples:

    >>> view.fourier_fk_domain(data, key = "src", index = 51, fmin = 0, fmax = 100)
    >>> view.fourier_fk_domain(data, key = "rec", index = 203, fmin = 0, fmax = 100)
    >>> view.fourier_fk_domain(data, key = "cmp", index = 315, fmin = 0, fmax = 100)
    >>> view.fourier_fk_domain(data, key = "off", index = 223750, fmin = 0, fmax = 100)
    '''    
    
    __check_keyword(key)
    __check_index(data, key, index)

    byte, label = __keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]
    
    sx = data.attributes(73)[traces] / data.attributes(71)[traces]
    sy = data.attributes(77)[traces] / data.attributes(71)[traces]    
    sz = data.attributes(45)[traces] / data.attributes(71)[traces]

    rx = data.attributes(81)[traces] / data.attributes(69)[traces]
    ry = data.attributes(85)[traces] / data.attributes(69)[traces]    
    rz = data.attributes(41)[traces] / data.attributes(69)[traces]
    
    distance = np.sqrt((sx - rx)**2 + (sy - ry)**2 + (sz - rz)**2)

    nx = len(traces)
    dh = np.median(np.abs(distance[1:] - distance[:-1])) 
    
    fk_seismic = np.fft.fftshift(np.fft.fft2(seismic))

    frequency = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    wavenumber = np.fft.fftshift(np.fft.fftfreq(nx, dh))

    mask = np.logical_and(frequency >= fmin, frequency <= fmax)

    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]

    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)

    floc = np.linspace(0, len(frequency[mask])-1, 11, dtype = int)
    flab = np.around(frequency[mask][floc][::-1], decimals = 1)

    kloc = np.linspace(0, len(wavenumber)-1, 5, dtype = int)
    klab = np.around(wavenumber[kloc], decimals = 3)

    scale = 0.9*np.std(seismic)
    
    fig, ax = plt.subplots(num = f"Common {label} gather with its 2D fourier transform", ncols = 2, nrows = 1, figsize = (10, 5))

    im = ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax[0].set_yticks(tloc)
    ax[0].set_yticklabels(tlab)
    ax[0].set_xticks(xloc)
    ax[0].set_xticklabels(xlab)

    ax[0].set_ylabel('Time [s]', fontsize = 15)
    ax[0].set_xlabel('Trace number', fontsize = 15)

    fk = ax[1].imshow(np.abs(fk_seismic[mask,:][::-1]), aspect = "auto", cmap = "jet")
    
    ax[1].set_yticks(floc)
    ax[1].set_yticklabels(flab)

    ax[1].set_xticks(kloc)
    ax[1].set_xticklabels(klab)

    ax[1].set_ylabel("Frequency [Hz]", fontsize = 15)
    ax[1].set_xlabel(r"Wavenumber [m$^{-1}$]", fontsize = 15)

    ax[1].set_ylabel("Frequency [Hz]")

    ax[0].cbar = fig.colorbar(im, ax = ax[0])
    ax[0].cbar.set_label("Amplitude", fontsize = 15) 

    ax[1].cbar = fig.colorbar(fk, ax = ax[1])
    ax[1].cbar.set_label("Amplitude", fontsize = 15) 
    
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

