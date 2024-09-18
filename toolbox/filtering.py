import scipy as sc
import numba as nb
import numpy as np
import segyio as sgy

import matplotlib.pyplot as plt

from toolbox import managing as mng

@nb.jit(nopython = True, parallel = True)
def radon_forward(seismic, dt, times, offsets, velocities, style):

    Nt = len(times)
    Nh = len(offsets)
    Np = len(velocities)

    domain = np.zeros((Nt, Nh, Np))

    if style == "linear":
        for tind in nb.prange(Nt): 
            for vind in nb.prange(Np):  
                for hind in nb.prange(Nh):
                    curvature = times[tind] + offsets[hind]/velocities[vind]                        
                    if 0 <= curvature <= (Nt-1)*dt:
                        domain[tind, hind, vind] = seismic[int(curvature/dt), hind]           

    elif style == "parabolic":
        for tind in nb.prange(Nt): 
            for vind in nb.prange(Np):  
                for hind in nb.prange(Nh):
                    curvature = times[tind] + (offsets[hind]/velocities[vind])**2
                    if 0 <= curvature <= (Nt-1)*dt:
                        domain[tind, hind, vind] = seismic[int(curvature/dt), hind]           
                
    elif style == "hyperbolic":
        for tind in nb.prange(Nt): 
            for vind in nb.prange(Np):  
                for hind in nb.prange(Nh):
                    curvature = np.sqrt(times[tind]**2 + (offsets[hind]/velocities[vind])**2)            
                    if 0 <= curvature <= (Nt-1)*dt:
                        domain[tind, hind, vind] = seismic[int(curvature/dt), hind]           

    return domain

@nb.jit(nopython = True, parallel = True)
def radon_adjoint(domain, dt, times, offsets, velocities, style):

    Nt = len(times)
    Nh = len(offsets)
    Np = len(velocities)

    data = np.zeros((Nt, Nh))

    if style == "linear":
        for tind in nb.prange(Nt): 
            for vind in nb.prange(Np):  
                for hind in nb.prange(Nh):
                    curvature = times[tind] + offsets[hind]/velocities[vind]        
                    if 0 <= curvature <= (Nt-1)*dt:
                        data[int(curvature/dt), hind] = domain[tind, hind, vind]           
                
    elif style == "parabolic":
        for tind in nb.prange(Nt): 
            for vind in nb.prange(Np):  
                for hind in nb.prange(Nh):
                    curvature = times[tind] + (offsets[hind]/velocities[vind])**2
                    if 0 <= curvature <= (Nt-1)*dt:
                        data[int(curvature/dt), hind] = domain[tind, hind, vind]           
                
    elif style == "hyperbolic":
        for tind in nb.prange(Nt): 
            for vind in nb.prange(Np):  
                for hind in nb.prange(Nh):
                    curvature = np.sqrt(times[tind]**2 + (offsets[hind]/velocities[vind])**2)            
                    if 0 <= curvature <= (Nt-1)*dt:
                        data[int(curvature/dt), hind] = domain[tind, hind, vind]           

    return data

def fourier_fx_domain(input : sgy.SegyFile, **kwargs) -> sgy.SegyFile:
    '''
    Fourier 1D bandpass filtering. This function apply this filter along all data traces, 
    exporting and returning a new data filtered. 
    
    ### Parameters:        
    
    data: segyio object.

    ### Optional parameters:
    
    key: header keyword options -> ['src', 'rec', 'off', 'cmp']
    
    index: integer that select a common gather.  

    fmin: minimum frequency.    
    
    fmax: maximum frequency.    
    
    output_name: path to export data results.    

    ### Returns:

    output: segyio object.

    ### Examples:

    >>> filt.fourier_fx_domain(data, fmax = 200)               # plots first shot             
    >>> filt.fourier_fx_domain(data, key = "rec", index = 789) # plots rec index 789  
    >>> filt.fourier_fx_domain(data, key = "cmp", index = 512) # plots cmp index 512
    '''    

    fmin = kwargs.get("fmin") if "fmin" in kwargs else 0.0
    fmax = kwargs.get("fmax") if "fmax" in kwargs else 50.0

    output_name = kwargs.get("output_name") if "output_name" in kwargs else f"data_filtered_{fmin}-{fmax}Hz.sgy"

    order = 5.0

    dt = input.attributes(117)[0][0] * 1e-6

    seismic_input = input.trace.raw[:].T

    seismic_bandpass = np.zeros_like(seismic_input)

    for i in range(input.tracecount):

        b, a = sc.signal.butter(order, [fmin, fmax], fs = 1/dt, btype = 'band')

        seismic_bandpass[:,i] = sc.signal.lfilter(b, a, seismic_input[:,i])

    sgy.tools.from_array2D(output_name, seismic_bandpass.T)
    
    output = sgy.open(output_name, "r+", ignore_geometry = True)
    output.header = input.header
    
    return output

def fourier_fk_domain(data: sgy.SegyFile, **kwargs) -> sgy.SegyFile:
    '''
    Plot the amplitude spectra of each trace in gather according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "cmp"]
    
    index: integer that select a common gather.  

    fmax: maximum frequency to visualize    
    
    ### Examples:

    >>> view.fourier_fx_domain(data, fmax = 200)               # plots first shot             
    >>> view.fourier_fx_domain(data, key = "rec", index = 789) # plots rec index 789  
    >>> view.fourier_fx_domain(data, key = "cmp", index = 512) # plots cmp index 512
    '''    

    fmax = kwargs.get("fmax") if "fmax" in kwargs else 100.0

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

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

    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]

    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)

    floc = np.linspace(0, len(frequency[mask])-1, 11, dtype = int)
    flab = np.around(np.ceil(frequency[mask][floc][::-1]), decimals = 1)

    kloc = np.linspace(0, len(wavenumber)-1, 5, dtype = int)
    klab = np.around(wavenumber[kloc], decimals = 3)

    scale = 0.8*np.std(seismic)
    
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 5))

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

    # current_clicks = []
    # polygons = []
    # polygon_paths = []

    # def on_click(event):
    #     global current_clicks
    #     if event.inaxes:
    #         current_clicks.append((event.xdata, event.ydata))
    #         ax.plot(event.xdata, event.ydata, 'ro')
    #         plt.draw()

    #         if len(current_clicks) >= 3:
    #             for artist in ax.patches:
    #                 artist.remove()

    #             for polygon in polygons:
    #                 ax.add_patch(polygon)

    #             polygon = Polygon(current_clicks, closed=True, edgecolor='black', facecolor='cyan', alpha=0.5)
    #             ax.add_patch(polygon)
    #             plt.draw()

    # def on_key(event):
    #     global current_clicks, polygons, polygon_paths
    #     if event.key == 'n':
    #         if len(current_clicks) >= 3:
    #             polygon = Polygon(current_clicks, closed=True, edgecolor='black', facecolor='cyan', alpha=0.5)
    #             polygons.append(polygon)
    #             polygon_path = Path(current_clicks)
    #             polygon_paths.append(polygon_path)

    #         current_clicks = []

    # fig, ax = plt.subplots()
    # ax.imshow(data, cmap='gray')
    # cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    # cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    # plt.show()

    # if polygon_paths:
    #     mask = np.zeros(data.shape)

    #     x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    #     xy = np.vstack((x.flatten(), y.flatten())).T

    #     for polygon_path in polygon_paths:
    #         inside = polygon_path.contains_points(xy).reshape(mask.shape)
    #         mask[inside] = 1

    #     mask = gaussian_filter(mask.astype(np.float32), sigma=7)

    #     masked_image = data * mask

    #     plt.figure()
    #     plt.imshow(masked_image, cmap='gray')
    #     plt.show()

def mute(data : sgy.SegyFile, velocity, t0, **kwargs) -> None:

    key = kwargs.get("key") if "key" in kwargs else "src"

    byte = mng.__keywords.get(key)

    if key == "cmp":
        _, cmps_per_traces = np.unique(data.attributes(byte)[:], return_counts = True)

        complete_cmp_indexes = np.where(cmps_per_traces == np.max(cmps_per_traces))[0]

        index = kwargs.get("index") if "index" in kwargs else complete_cmp_indexes[0]
    else:
        index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0]

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    traces = np.where(data.attributes(byte)[:] == index)[0]
    #offset = data.attributes(37)[traces] / data.attributes(69)[traces]
    offset = np.arange(len(traces)) * 15
   
    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    t = t0 + np.abs((offset) / velocity)

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]
    
    seismic_mute = np.copy(seismic)

    for i in range(int(len(offset))):
        limit = int(t[i] / dt)
        seismic_mute[:limit, i] = 0  

    scale = 0.8*np.std(seismic)
    scale_mute = 0.8*np.std(seismic_mute)

    fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (15, 5))
    
    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]
    tloc = np.linspace(0, nt, 11, dtype = int)
    tlab = np.around(tloc * dt, decimals = 1)
    
    img = ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
    ax[0].set_xticks(xloc)
    ax[0].set_xticklabels(xlab)
    ax[0].set_yticks(tloc)
    ax[0].set_yticklabels(tlab)
    ax[0].set_ylabel('Time [s]', fontsize = 15)
    ax[0].set_xlabel('Trace number', fontsize = 15)

    # cbar = fig.colorbar(img, ax = ax)
    # cbar.set_label("Amplitude", fontsize = 15)
    
    ax[1].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale, extent=[0, len(traces), nt*dt,0])
    ax[1].plot(np.arange(len(offset)), t, 'or')
    #ax[1].plot(offset, t, 'red')

    ax[1].set_xticks(xloc)
    ax[1].set_xticklabels(xlab)
    # ax[1].set_yticks(tloc)
    # ax[1].set_yticklabels(tlab)
    ax[1].set_ylabel('Time [s]', fontsize = 15)
    ax[1].set_xlabel('Trace number', fontsize = 15)

    ax[2].imshow(seismic_mute, aspect = "auto", cmap = "Greys", vmin = -scale_mute, vmax = scale_mute)
    ax[2].set_xticks(xloc)
    ax[2].set_xticklabels(xlab)
    ax[2].set_yticks(tloc)
    ax[2].set_yticklabels(tlab)
    ax[2].set_ylabel('Time [s]', fontsize = 15)
    ax[2].set_xlabel('Trace number', fontsize = 15)

    fig.tight_layout()
    plt.show()
