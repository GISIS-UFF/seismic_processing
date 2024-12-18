import scipy as sc
import numba as nb
import numpy as np
import segyio as sgy

import matplotlib.pyplot as plt

from toolbox import managing as mng

@nb.jit(nopython = True, parallel = True)
def radon_forward(seismic : np.ndarray, dt : float, times : np.ndarray, offsets : np.ndarray, velocities : np.ndarray, style : str) -> np.ndarray:

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
def radon_adjoint(domain : np.ndarray, dt : float, times : np.ndarray, offsets : np.ndarray, velocities : np.ndarray, style : str) -> np.ndarray:

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

def fourier_fx_domain(data_input : sgy.SegyFile, **kwargs) -> sgy.SegyFile:
    '''
    Fourier 1D bandpass filtering. This function apply this filter along all data traces, 
    exporting and returning a new data filtered. 
    
    ### Parameters:        
    
    data_input: segyio object.

    ### Optional parameters:
    
    fmin: minimum frequency.    
    
    fmax: maximum frequency.    
    
    output_name: path to export data results.    

    ### Returns:

    data_output: segyio object.

    ### Examples:

    >>> filt.fourier_fx_domain(data, fmax = 200)
    >>> filt.fourier_fx_domain(data, fmin = 10, fmax = 100)            
    '''    

    fmin = kwargs.get("fmin") if "fmin" in kwargs else 0.0
    fmax = kwargs.get("fmax") if "fmax" in kwargs else 50.0

    output_name = kwargs.get("output_name") if "output_name" in kwargs else f"data_bandpass.sgy"

    order = 5.0

    dt = data_input.attributes(117)[0][0] * 1e-6

    seismic_input = data_input.trace.raw[:].T

    seismic_bandpass = np.zeros_like(seismic_input)

    for i in range(data_input.tracecount):

        b, a = sc.signal.butter(order, [fmin, fmax], fs = 1/dt, btype = 'band')

        seismic_bandpass[:,i] = sc.signal.lfilter(b, a, seismic_input[:,i])

    sgy.tools.from_array2D(output_name, seismic_bandpass.T)
    
    data_output = sgy.open(output_name, "r+", ignore_geometry = True)
    data_output.header = data_input.header
    
    return data_output

def fourier_fk_domain(data: sgy.SegyFile, **kwargs) -> sgy.SegyFile:
    '''
    Fourier 1D bandpass filtering. This function apply this filter along all data traces, 
    exporting and returning a new data filtered. 
    
    ### Parameters:        
    
    data_input: segyio object.

    ### Optional parameters:
        
    fmax: maximum frequency for visualization.    
    
    output_name: path to export data results.    

    ### Returns:

    data_output: segyio object.

    ### Examples:

    >>> filt.fourier_fx_domain(data, fmax = 200)
    >>> filt.fourier_fx_domain(data, fmin = 10, fmax = 100)            
    '''    

    key = "cmp"
    byte = mng.__keywords.get(key)
    
    fmax = kwargs.get("fmax") if "fmax" in kwargs else 100.0

    cmp_indexes = mng.get_keyword_indexes(data, key)
    _, cmps_per_traces = np.unique(data.attributes(byte)[:], return_counts = True)
    complete_cmp_indexes = np.where(cmps_per_traces > np.max(cmps_per_traces))[0]
    indexes = cmp_indexes[complete_cmp_indexes[:]]

    print(indexes)


    # index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

    # mng.__check_keyword(key)
    # mng.__check_index(data, key, index)

    # byte = mng.__keywords.get(key)

    # traces = np.where(data.attributes(byte)[:] == index)[0]

    # nt = data.attributes(115)[0][0]
    # dt = data.attributes(117)[0][0] * 1e-6

    # seismic = data.trace.raw[:].T
    # seismic = seismic[:, traces]
        
    # distance = data.attributes(37)[traces] / data.attributes(69)[traces]

    # nx = len(traces)
    # dh = np.median(np.abs(np.abs(distance[1:]) - np.abs(distance[:-1]))) 

    # fk_seismic = np.fft.fftshift(np.fft.fft2(seismic))

    # frequency = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    # wavenumber = np.fft.fftshift(np.fft.fftfreq(nx, dh))

    # mask = np.logical_and(frequency >= 0.0, frequency <= fmax)

    # xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    # xlab = traces[xloc]

    # tloc = np.linspace(0, nt-1, 11, dtype = int)
    # tlab = np.around(tloc*dt, decimals = 1)

    # floc = np.linspace(0, len(frequency[mask])-1, 11, dtype = int)
    # flab = np.around(np.ceil(frequency[mask][floc][::-1]), decimals = 1)

    # kloc = np.linspace(0, len(wavenumber)-1, 5, dtype = int)
    # klab = np.around(wavenumber[kloc], decimals = 3)

    # scale = 0.8*np.std(seismic)
    
    # fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 5))

    # im = ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    # ax[0].set_yticks(tloc)
    # ax[0].set_yticklabels(tlab)
    # ax[0].set_xticks(xloc)
    # ax[0].set_xticklabels(xlab)

    # ax[0].set_ylabel('Time [s]', fontsize = 15)
    # ax[0].set_xlabel('Trace number', fontsize = 15)

    # fk = ax[1].imshow(np.abs(fk_seismic[mask,:][::-1]), aspect = "auto", cmap = "jet")
    
    # ax[1].set_yticks(floc)
    # ax[1].set_yticklabels(flab)

    # ax[1].set_xticks(kloc)
    # ax[1].set_xticklabels(klab)

    # ax[1].set_ylabel("Frequency [Hz]", fontsize = 15)
    # ax[1].set_xlabel(r"Wavenumber [m$^{-1}$]", fontsize = 15)

    # ax[1].set_ylabel("Frequency [Hz]")

    # ax[0].cbar = fig.colorbar(im, ax = ax[0])
    # ax[0].cbar.set_label("Amplitude", fontsize = 15) 

    # ax[1].cbar = fig.colorbar(fk, ax = ax[1])
    # ax[1].cbar.set_label("Amplitude", fontsize = 15) 
    
    # fig.tight_layout()
    # plt.show()

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

def mute(data_input : sgy.SegyFile, **kwargs) -> None:
    '''
    Mute traces according with a linear function : t = ti + x / v.

    x -> offset according to segy data (it's not a parameter).
     
    ### Parameters:        
    
    data: segyio object.

    ### Optional parameters:
    
    time (ti): initial time to delay the linear mute in seconds.    
    
    velocity (v): defines slope of the linear mute in meters per seconds.    
    
    output_name: path to export data results.    

    ### Returns:

    output: segyio object.

    ### Examples:

    >>> filt.mute(data, velocity = 1000)
    >>> filt.mute(data, time = 0.5, velocity = 1200)            
    '''    

    time = kwargs.get("time") if "time" in kwargs else 0.0 
    velocity = kwargs.get("velocity") if "velocity" in kwargs else 1500.0
    
    output_name = kwargs.get("output_name") if "output_name" in kwargs else f"data_muted.sgy"

    dt = data_input.attributes(117)[0][0] * 1e-6

    seismic_input = data_input.trace.raw[:].T

    offset = data_input.attributes(37)[:] / data_input.attributes(69)[:]

    tmute = np.array((time + np.abs(offset / velocity)) / dt, dtype = int)
        
    for i in range(data_input.tracecount):
        seismic_input[:tmute[i], i] *= 0.0

    sgy.tools.from_array2D(output_name, seismic_input.T)
    
    data_output = sgy.open(output_name, "r+", ignore_geometry = True)
    data_output.header = data_input.header
    
    return data_output

def apply_agc(data: sgy.SegyFile , time_window : float, **kwargs):
   
    #TODO: add docstring

    # time_window in seconds



    key = kwargs.get("key") if "key" in kwargs else "src"

    byte = mng.__keywords.get(key)

    if key == "cmp":      
        cmp_indexes = mng.get_keyword_indexes(data, key)
        _, cmps_per_traces = np.unique(data.attributes(byte)[:], return_counts = True)
        complete_cmp_indexes = np.where(cmps_per_traces == np.max(cmps_per_traces))[0]
        index = kwargs.get("index") if "index" in kwargs else cmp_indexes[complete_cmp_indexes[0]]

    elif key == "rec":
        rec_indexes = mng.get_keyword_indexes(data, key)
        _, src_per_rec = np.unique(data.attributes(byte)[:], return_counts = True)
        complete_rec_indexes = np.where(src_per_rec == np.max(src_per_rec))[0]
        index = kwargs.get("index") if "index" in kwargs else rec_indexes[complete_rec_indexes[0]]
    else: 
        index = kwargs.get("index") if "index" in kwargs else mng.get_keyword_indexes(data, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    seismic = np.zeros((nt, len(traces)))

    for i in range(len(traces)):
        seismic[:,i] = data.trace.raw[traces[i]] 

    sliding_window = int(time_window / dt) + 1

    for i in range(len(traces)):
        trace = seismic[:, i]
        l, h = 0, sliding_window - 1
        mid = (l + h) // 2
        while h < nt - 1:
            window_samples = trace[l:h]
            mean_amplitude = np.mean(np.abs(window_samples))
            trace[mid] /= mean_amplitude + 1e-6

            l, h, mid = l + 1, h + 1, mid + 1

        seismic[:,i] = trace

    return seismic


def time_variant_filtering(data : sgy.SegyFile, num_windows: int, **kwargs) -> None:
    """
    Apply time-variant filtering to normalize the amplitude of each trace in a 2D seismic dataset.
    Each trace (row) is processed independently.

    Parameters:
    - data (numpy.ndarray): 2D array representing the seismic dataset (traces as rows).
    - dt (float): Time sampling interval.
    - num_windows (int): Number of windows to divide each trace for normalization.

    Returns:
    - numpy.ndarray: The scaled 2D dataset after normalization.
    """
    key = kwargs.get("key") if "key" in kwargs else "src"

    byte = mng.__keywords.get(key)

    if key == "cmp":      
        cmp_indexes = mng.get_keyword_indexes(data, key)
        _, cmps_per_traces = np.unique(data.attributes(byte)[:], return_counts = True)
        complete_cmp_indexes = np.where(cmps_per_traces == np.max(cmps_per_traces))[0]
        index = kwargs.get("index") if "index" in kwargs else cmp_indexes[complete_cmp_indexes[0]]

    elif key == "rec":
        rec_indexes = mng.get_keyword_indexes(data, key)
        _, src_per_rec = np.unique(data.attributes(byte)[:], return_counts = True)
        complete_rec_indexes = np.where(src_per_rec == np.max(src_per_rec))[0]
        index = kwargs.get("index") if "index" in kwargs else rec_indexes[complete_rec_indexes[0]]
    
    else:    
        index = kwargs.get("index") if "index" in kwargs else mng.get_keyword_indexes(data, key)[0] 


    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    seismic = np.zeros((nt, len(traces)))

    for i in range(len(traces)):
        seismic[:,i] = data.trace.raw[traces[i]] 

    seismic = seismic.T

    if seismic.ndim != 2:
        raise ValueError("seismic must be a 2D array.")
    if num_windows <= 0:
        raise ValueError("num_windows must be greater than 0.")

    seismic_filt = np.zeros_like(seismic)

    for i in range(seismic.shape[0]): 
        trace = seismic[i, :]
        normalization_start_time = (np.argmax(trace) + 100) * dt  
        normalization_start_time = int(normalization_start_time / dt)
        
        if normalization_start_time >= len(trace):
            raise ValueError("normalization_start_time exceeds the length of the signal.")

        window_size = (len(trace) - normalization_start_time) // num_windows
        max_scale = np.max(trace)
        min_scale = np.min(trace)  

        signal_scaled = np.copy(trace)
        for j in range(num_windows):
            start_idx = normalization_start_time + j * window_size
            end_idx = start_idx + window_size if j < num_windows - 1 else len(trace)
            signal_max = np.max(np.abs(trace[start_idx:end_idx]))  

            if signal_max > 1e-6:  
                normalized_segment = trace[start_idx:end_idx] / signal_max
                normalized_segment = np.where(
                    normalized_segment > 0,
                    normalized_segment * max_scale, 
                    normalized_segment * abs(min_scale) 
                )
                signal_scaled[start_idx:end_idx] = normalized_segment

        seismic_filt[i, :] = signal_scaled


    # -------------------------------------------------------------------------------------------
    # -------------------Plot -------------------------------------------------------------------

    scale = 0.9 * np.std(seismic)

    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 5)) 

    im1 = ax[0].imshow(seismic.T, aspect = 'auto', cmap = 'Greys', vmin = -scale, vmax = scale) 

    ax[0].set_xlabel('Relative trace number', fontsize = 15) 
    ax[0].set_ylabel('Time [s]', fontsize = 15)
    cbar1 = fig.colorbar(im1, ax = ax[0])
    cbar1.set_label("Amplitude", fontsize = 15)

    im2 = ax[1].imshow(seismic_filt.T, aspect = 'auto', cmap = 'Greys', vmin = -scale, vmax = scale)

    ax[1].set_xlabel('Velocity [m/s]', fontsize = 15) 
    ax[1].set_ylabel('Time [s]', fontsize = 15)
    ax[1].set_xlabel('Relative trace number', fontsize = 15) 
    ax[1].set_ylabel('Time [s]', fontsize = 15)
    cbar2 = fig.colorbar(im2, ax = ax[1])
    cbar2.set_label("Amplitude", fontsize = 10)

    fig.tight_layout() 
    plt.show()


