import numpy as np
import segyio as sgy

import matplotlib.pyplot as plt

from toolbox import managing as mng
from toolbox import filtering as filt

def gather(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot a prestack seismic gather according to a specific header keyword.

    ### Mandatory Parameters:

    data: segyio object.

    ### Other Parameters:

    key: header keyword options -> ["src", "rec", "off", "cmp"] - "src" as default
    
    index: integer that select a common gather. 
    
    ### Hints:
    
    For each header keyword the first index is plotted. 
    
    Specially for a cmp and rec gather, the first complete data is plotted. 
        
    ### Examples:

    >>> view.gather(data)                            
    >>> view.gather(data, key = "off")               
    >>> view.gather(data, key = "rec", index = 789) 
    >>> view.gather(data, key = "cmp", index = 512) 
    '''    
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
    
    scale = 0.9*np.std(seismic)

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (10, 5))

    img = ax.imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)

    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc * dt, decimals = 3)
    
    ax.set_xticks(xloc)
    ax.set_xticklabels(xloc)
    
    ax.set_yticks(tloc)
    ax.set_yticklabels(tlab)

    ax.set_ylabel('Time [s]', fontsize = 15)
    ax.set_xlabel('Relative trace number', fontsize = 15)

    cbar = fig.colorbar(img, ax = ax)
    cbar.set_label("Amplitude", fontsize = 15)

    fig.tight_layout()
    plt.show()

def geometry(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot geometry, cmp coodinates and current configuration 
    according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    ### Other Parameters:

    key: header keyword options -> ["src", "rec", "off", "cmp"] - "src" as default
    
    index: integer that select a common gather. 
    
    ### Hints:
    
    For each header keyword the first index is plotted. 
    
    Specially for a cmp and rec gather, the first complete data is plotted. 

    ### Examples:

    >>> view.geometry(data)                           
    >>> view.geometry(data, key = "off")              
    >>> view.geometry(data, key = "rec", index = 35)  
    >>> view.geometry(data, key = "cmp", index = 512) 
    '''    

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

    sx_complete = data.attributes(73)[:] / data.attributes(69)[:] 
    sy_complete = data.attributes(77)[:] / data.attributes(69)[:]    

    rx_complete = data.attributes(81)[:] / data.attributes(69)[:]
    ry_complete = data.attributes(85)[:] / data.attributes(69)[:]

    cmpx = data.attributes(181)[:] / data.attributes(69)[:] 
    cmpy = data.attributes(185)[:] / data.attributes(69)[:]

    xmin = min(np.min(sx_complete), np.min(rx_complete)) - 100
    xmax = max(np.max(sx_complete), np.max(rx_complete)) + 100

    ymin = min(np.min(sy_complete), np.min(ry_complete)) - 100
    ymax = max(np.max(sy_complete), np.max(ry_complete)) + 100

    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (15, 5))

    def set_config(p):
        ax[p].set_xlabel("X [m]", fontsize = 15)
        ax[p].set_ylabel("y [m]", fontsize = 15)
        ax[p].set_xlim([xmin, xmax])
        ax[p].set_ylim([ymin, ymax])
        ax[p].legend(loc = "lower left")

    ax[0].plot(rx_complete, ry_complete, ".", label = "Receivers")
    ax[0].plot(sx_complete, sy_complete, ".", label = "Sources")
    ax[0].set_title("Complete geometry", fontsize = 15)
    set_config(0)

    ax[1].plot(rx_complete[traces], ry_complete[traces], ".", label="Receivers")
    ax[1].plot(sx_complete[traces], sy_complete[traces], ".", label="Sources")
    ax[1].plot(cmpx[traces], cmpy[traces], ".", label = "CMPs")
    ax[1].set_title("Local geometry", fontsize = 15)
    set_config(1)

    fig.tight_layout()
    plt.show()

def fourier_fx_domain(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plots the amplitude spectra for each trace in gather according to a specific header keyword. 
    Also plots a trace with its amplitude spectra selected by user.
    
    ### Parameters:        
    
    data: segyio object.

    ### Optional parameters:
                    
    key: header keyword options -> ['src', 'rec', 'cmp'].
    
    index: integer that select a common gather.  
    
    fmax: maximum frequency to visualize.    

    trace_number: relative trace according with current gather to show individually.

    ### Hints:
    
    For each header keyword the first index is plotted. 
    
    Specially for a cmp and rec gather, the first complete data is plotted. 

    ### Examples:

    >>> view.fourier_fx_domain(data, trace_number = 100)                    
    >>> view.fourier_fx_domain(data, key = "off", fmax = 200)  
    >>> view.fourier_fx_domain(data, key = "rec", index = 789)   
    >>> view.fourier_fx_domain(data, key = "cmp", index = 512) 
    '''    

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

    fmax = kwargs.get("fmax") if "fmax" in kwargs else 100.0
    trace_number = kwargs.get("trace_number") if "trace_number" in kwargs else 0

    traces = np.where(data.attributes(byte)[:] == index)[0]    

    if trace_number < 0 or trace_number > len(traces):
        print("Wrong argument for trace_number!")
        print(f"Relative traces available: 0 to {len(traces)-1}")
        exit()

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6
    
    seismic = np.zeros((nt, len(traces)))

    for i in range(len(traces)):
        seismic[:,i] = data.trace.raw[traces[i]] 

    time = np.arange(nt)*dt
    frequency = np.fft.fftfreq(nt, dt)
    fx_seismic = np.fft.fft(seismic, axis = 0)

    for i in range(len(traces)):
        fx_seismic[:,i] *= 1.0 / np.max(fx_seismic[:,i]) 

    scale = 0.9*np.std(seismic)

    mask = np.logical_and(frequency >= 0.0, frequency <= fmax)

    floc = np.linspace(0, len(frequency[mask]), 11, dtype = int)
    flab = np.around(np.ceil(frequency[floc]), decimals = 1)
    
    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    
    tloc = np.linspace(0, nt, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)

    fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (10, 9))

    im = ax[0,0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax[0,0].plot(trace_number*np.ones(nt), time/dt, "--r")
    ax[0,0].set_xticks(xloc)
    ax[0,0].set_yticks(tloc)
    ax[0,0].set_xticklabels(xloc)
    ax[0,0].set_yticklabels(tlab)
    ax[0,0].set_ylabel('Time [s]', fontsize = 15)
    ax[0,0].set_xlabel('Relative trace number', fontsize = 15)
    ax[0,0].cbar = fig.colorbar(im, ax = ax[0,0])
    ax[0,0].cbar.set_label("Amplitude", fontsize = 15) 

    ax[0,1].plot(seismic[:, trace_number], time)
    ax[0,1].set_xlabel("Amplitude", fontsize = 15)
    ax[0,1].set_ylabel("Time [s]", fontsize = 15)
    ax[0,1].set_xlim([-5*scale, 5*scale])
    ax[0,1].invert_yaxis()

    fx = ax[1,0].imshow(np.abs(fx_seismic[mask,:]), aspect = "auto", cmap = "jet")

    ax[1,0].plot(trace_number*np.ones(len(frequency[mask])), np.arange(len(frequency[mask])), "--r")
    ax[1,0].set_xticks(xloc)
    ax[1,0].set_yticks(floc)
    ax[1,0].set_xticklabels(xloc)
    ax[1,0].set_yticklabels(flab)
    ax[1,0].set_ylabel("Frequency [Hz]", fontsize = 15)
    ax[1,0].set_xlabel("Relative trace number", fontsize = 15)
    ax[1,0].cbar = fig.colorbar(fx, ax = ax[1,0])
    ax[1,0].cbar.set_label("Amplitude", fontsize = 15) 

    ax[1,1].plot(np.abs(fx_seismic[mask, trace_number]), frequency[mask])
    ax[1,1].set_xlabel("Normalized amplitude", fontsize = 15)
    ax[1,1].set_ylabel("Frequency [Hz]", fontsize = 15)
    ax[1,1].invert_yaxis()

    fig.tight_layout()
    plt.show()

def fourier_fk_domain(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plots the 2D fourier transform according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    ### Optional parameters:
                    
    key: header keyword options -> ['src', 'rec', 'cmp'].
    
    index: integer that select a common gather.  

    fmax: maximum frequency to visualize.    

    ### Hints:
    
    For each header keyword the first index is plotted. 
    
    Specially for a cmp and rec gather, the first complete data is plotted. 
        
    ### Examples:

    >>> view.fourier_fk_domain(data, fmax = 200)                            
    >>> view.fourier_fk_domain(data, key = "off", fmax = 200)  
    >>> view.fourier_fk_domain(data, key = "rec", index = 789)   
    >>> view.fourier_fk_domain(data, key = "cmp", index = 512) 
    '''    

    fmax = kwargs.get("fmax") if "fmax" in kwargs else 100.0

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
        
    dist = data.attributes(37)[traces] / data.attributes(69)[traces]
    
    dh = np.abs(dist[0]) if key == "off" else np.median(np.abs(np.abs(dist[1:]) - np.abs(dist[:-1])))

    fk_seismic = np.fft.fftshift(np.fft.fft2(seismic))

    frequency = np.fft.fftshift(np.fft.fftfreq(nt, dt))
    wavenumber = np.fft.fftshift(np.fft.fftfreq(len(traces), dh))

    mask = np.logical_and(frequency >= 0.0, frequency <= fmax)

    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)

    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)

    floc = np.linspace(0, len(frequency[mask])-1, 11, dtype = int)
    flab = np.around(np.ceil(frequency[mask][floc][::-1]), decimals = 1)

    kloc = np.linspace(0, len(wavenumber)-1, 5, dtype = int)
    klab = np.around(wavenumber[kloc], decimals = 5)

    scale = 0.9*np.std(seismic)
    
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 5))

    im = ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax[0].set_xticks(xloc)
    ax[0].set_yticks(tloc)
    ax[0].set_xticklabels(xloc)
    ax[0].set_yticklabels(tlab)
    ax[0].set_ylabel('Time [s]', fontsize = 15)
    ax[0].set_xlabel('Relative trace number', fontsize = 15)
    ax[0].cbar = fig.colorbar(im, ax = ax[0])
    ax[0].cbar.set_label("Amplitude", fontsize = 15) 

    fk = ax[1].imshow(np.abs(fk_seismic[mask,:][::-1]), aspect = "auto", cmap = "jet")
    
    ax[1].set_xticks(kloc)
    ax[1].set_yticks(floc)
    ax[1].set_xticklabels(klab)
    ax[1].set_yticklabels(flab)
    ax[1].set_ylabel("Frequency [Hz]", fontsize = 15)
    ax[1].set_xlabel(r"Wavenumber [m$^{-1}$]", fontsize = 15)
    ax[1].cbar = fig.colorbar(fk, ax = ax[1])
    ax[1].cbar.set_label("Amplitude", fontsize = 15) 
    
    fig.tight_layout()
    plt.show()

def radon_transform(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plots the Radon transform according to a specific header keyword.
    
    ### Parameters:        
    
    input: segyio object.

    ### Optional parameters:
                    
    key: header keyword options -> ['src', 'rec', 'cmp'].

    index: integer that select a common gather.  

    style: Radon transform type -> ['linear', 'parabolic', 'hyperbolic'].    

    vmin: Minimum velocity (commonly 1000 m/s).

    vmax: Maximum velocity (commonly 3000 m/s).

    ### Hints:
    
    For each header keyword the first index is plotted. 
    
    Specially for a cmp and rec gather, the first complete data is plotted. 
    
    ### Examples:
    
    >>> view.radon_transform(data, style = "linear")
    >>> view.radon_transform(data, key = "rec", style = "hyperbolic")
    >>> view.radon_transform(data, key = "cmp", vmin = 100, vmax = 1000)
    '''  

    vmin = kwargs.get("vmin") if "vmin" in kwargs else 1000 
    vmax = kwargs.get("vmax") if "vmax" in kwargs else 3000 
    nvel = kwargs.get("nvel") if "nvel" in kwargs else 101

    style = kwargs.get("style").lower() if "style" in kwargs else "parabolic"

    if style not in ['linear', 'parabolic', 'hyperbolic']:
        print(f"Error: {style} style is not defined! Use a valid style: ['linear', 'parabolic', 'hyperbolic']")
        exit()

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

    elif key == "off": 
        print("Wrong argument for key!")
        print("Possible keys: ['src', 'rec', 'cmp']")
        exit()
    
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

    times = np.arange(nt) * dt
    offsets = data.attributes(37)[traces] / data.attributes(69)[traces]
    velocities = np.linspace(vmin, vmax, nvel)

    domain = filt.radon_forward(seismic, dt, times, offsets, velocities, style)

    radon = np.sum(domain, axis = 1)
    
    scale1 = 0.9*np.std(seismic)
    scale2 = 0.9*np.std(radon)

    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)

    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc * dt, decimals = 3)
    
    vloc = np.linspace(0, nvel-1, 5, dtype = int)
    vlab = velocities[vloc]

    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 5)) 

    im1 = ax[0].imshow(seismic, aspect = 'auto', cmap = 'Greys', vmin = -scale1, vmax = scale1) 
    ax[0].set_xticks(xloc)
    ax[0].set_yticks(tloc)
    ax[0].set_xticklabels(xloc)
    ax[0].set_yticklabels(tlab)
    ax[0].set_xlabel('Relative trace index', fontsize = 15) 
    ax[0].set_ylabel('Time [s]', fontsize = 15)
    cbar1 = fig.colorbar(im1, ax = ax[0])
    cbar1.set_label("Amplitude", fontsize = 15)

    im2 = ax[1].imshow(radon, aspect = 'auto', cmap = 'jet', vmin = -scale2, vmax = scale2)

    ax[1].set_xlabel('Velocity [m/s]', fontsize = 15) 
    ax[1].set_ylabel('Time [s]', fontsize = 15)
    ax[1].set_xticks(vloc)
    ax[1].set_yticks(tloc)
    ax[1].set_xticklabels(vlab)
    ax[1].set_yticklabels(tlab)
    ax[1].set_xlabel('Velocity [m/s]', fontsize = 15) 
    ax[1].set_ylabel('Time [s]', fontsize = 15)
    cbar2 = fig.colorbar(im2, ax = ax[1])
    cbar2.set_label("Amplitude", fontsize = 10)

    fig.tight_layout() 
    plt.show()

def difference(input : sgy.SegyFile, output : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot a difference between prestack seismic gathers according to a specific header keyword
    
    ### Parameters:        
    
    input: segyio object.
    
    output: segyio object.

    ### Optional parameters:

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    ### Hints:
    
    For each header keyword the first index is plotted. 
    
    Specially for a cmp and rec gather, the first complete data is plotted. 
        
    ### Examples:

    >>> view.difference(data, data_filt, key = "src")
    >>> view.difference(data, data_filt, key = "rec")
    >>> view.difference(data, data_filt, key = "cmp")
    >>> view.difference(data, data_filt, key = "off")
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"

    byte = mng.__keywords.get(key)

    if key == "cmp":      
        cmp_indexes = mng.get_keyword_indexes(input, key)
        _, cmps_per_traces = np.unique(input.attributes(byte)[:], return_counts = True)
        complete_cmp_indexes = np.where(cmps_per_traces == np.max(cmps_per_traces))[0]
        index = kwargs.get("index") if "index" in kwargs else cmp_indexes[complete_cmp_indexes[0]]

    elif key == "rec":
        rec_indexes = mng.get_keyword_indexes(input, key)
        _, src_per_rec = np.unique(input.attributes(byte)[:], return_counts = True)
        complete_rec_indexes = np.where(src_per_rec == np.max(src_per_rec))[0]
        index = kwargs.get("index") if "index" in kwargs else rec_indexes[complete_rec_indexes[0]]

    elif key == "off": 
        print("Wrong argument for key!")
        print("Possible keys: ['src', 'rec', 'cmp']")
        exit()
    
    else:    
        index = kwargs.get("index") if "index" in kwargs else mng.get_keyword_indexes(input, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(input, key, index)

    traces = np.where(input.attributes(byte)[:] == index)[0]
    
    nt = input.attributes(115)[0][0]
    dt = input.attributes(117)[0][0] * 1e-6

    seismic_input = np.zeros((nt, len(traces)))
    seismic_output = np.zeros((nt, len(traces)))

    for i in range(len(traces)):
        seismic_input[:,i] = input.trace.raw[traces[i]] 
        seismic_output[:,i] = output.trace.raw[traces[i]] 

    seismic_diff = seismic_output - seismic_input

    scale = 0.9*np.std(seismic_input)

    fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (18, 5))

    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 3)

    def set_config(p, fx):
                        
        ax[p].set_yticks(tloc)
        ax[p].set_yticklabels(tlab)
        ax[p].set_xticks(xloc)
        ax[p].set_xticklabels(xloc)
        
        ax[p].set_xlabel('Relative trace number', fontsize = 15)
        ax[p].set_ylabel('Time [s]', fontsize = 15)

        ax[p].cbar = fig.colorbar(fx, ax = ax[p])
        ax[p].cbar.set_label("Amplitude", fontsize = 15) 

    fx = ax[0].imshow(seismic_input, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
    set_config(0, fx)

    fx = ax[1].imshow(seismic_output, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
    set_config(1, fx)

    fx = ax[2].imshow(seismic_diff, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
    set_config(2, fx)
        
    fig.tight_layout()
    plt.show()    
