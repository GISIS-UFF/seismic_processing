import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline



ponto_parada= None
interpolated_line = None


from numba import jit

from toolbox import managing as mng

def gather(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot a prestack seismic gather according to a specific header keyword.

    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  
        
    ### Examples:

    >>> view.gather(data)                           # plots first shot 
    >>> view.gather(data, key = "off")              # plots first offset 
    >>> view.gather(data, key = "rec", index = 789) # plots rec index 789
    >>> view.gather(data, key = "cmp", index = 512) # plots cmp index 512
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte = mng.__keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    scale = 0.8*np.std(seismic)

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (10, 5))

    img = ax.imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]

    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc * dt, decimals = 3)
    
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

def geometry(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot geometry, cmp coodinates the current configuration according 
    to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    ### Examples:

    >>> view.geometry(data)                           # plots first shot 
    >>> view.geometry(data, key = "off")              # plots first offset
    >>> view.geometry(data, key = "rec", index = 789) # plots rec index 789
    >>> view.geometry(data, key = "cmp", index = 512) # plots cmp index 512
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte = mng.__keywords.get(key)

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

    fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (15, 5))

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

    ax[1].plot(cmpx, cmpy, label = "CMP")
    ax[1].set_title("Coverage", fontsize = 15)
    set_config(1)

    ax[2].plot(rx_complete[traces], ry_complete[traces], ".", label="Receivers")
    ax[2].plot(sx_complete[traces], sy_complete[traces], ".", label="Sources")
    ax[2].set_title("Local geometry", fontsize = 15)
    set_config(2)

    fig.tight_layout()
    plt.show()

def fourier_fx_domain(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot the amplitude spectra of each trace in gather according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  
    
    fmax: maximum frequency to visualize.    

    trace_number: relative trace to show individually.

    ### Examples:

    >>> view.fourier_fx_domain(data, trace_number = 100)       # plots first shot             
    >>> view.fourier_fx_domain(data, key = "off", fmax = 200)  # plots first offset
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

    trace_number = kwargs.get("trace_number") if "trace_number" in kwargs else 0

    if trace_number > len(traces):
        print("Wrong argument for trace_number!")
        print(f"Relative traces available: 0 to {len(traces)-1}")
        exit()

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6
    
    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    time = np.arange(nt)*dt
    frequency = np.fft.fftfreq(nt, dt)
    fx_seismic = np.fft.fft(seismic, axis = 0)

    for i in range(len(traces)):
        fx_seismic[:,i] *= 1.0 / np.max(fx_seismic[:,i]) 

    scale = 0.8*np.std(seismic)

    mask = np.logical_and(frequency >= 0.0, frequency <= fmax)

    floc = np.linspace(0, len(frequency[mask])-1, 11, dtype = int)
    flab = np.around(np.ceil(frequency[floc]), decimals = 1)
    
    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]
    
    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)

    fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize = (10, 9))

    im = ax[0,0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    ax[0,0].plot(trace_number*np.ones(nt), time/dt, "--r")

    ax[0,0].set_yticks(tloc)
    ax[0,0].set_yticklabels(tlab)
    ax[0,0].set_xticks(xloc)
    ax[0,0].set_xticklabels(xlab)

    ax[0,0].set_ylabel('Time [s]', fontsize = 15)
    ax[0,0].set_xlabel('Trace number', fontsize = 15)

    ax[0,0].cbar = fig.colorbar(im, ax = ax[0,0])
    ax[0,0].cbar.set_label("Amplitude", fontsize = 15) 


    ax[0,1].plot(seismic[:, trace_number], time)
    ax[0,1].set_xlabel("Amplitude", fontsize = 15)
    ax[0,1].set_ylabel("Time [s]", fontsize = 15)
    ax[0,1].set_xlim([-5*scale, 5*scale])
    ax[0,1].invert_yaxis()


    fx = ax[1,0].imshow(np.abs(fx_seismic[mask,:]), aspect = "auto", cmap = "jet")

    ax[1,0].plot(trace_number*np.ones(len(frequency[mask])), np.arange(len(frequency[mask])), "--r")

    ax[1,0].set_yticks(floc)
    ax[1,0].set_yticklabels(flab)
    ax[1,0].set_xticks(xloc)
    ax[1,0].set_xticklabels(xlab)

    ax[1,0].set_ylabel("Frequency [Hz]", fontsize = 15)
    ax[1,0].set_xlabel("Trace number", fontsize = 15)

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

def difference(input : sgy.SegyFile, output : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot a difference between prestack seismic gathers according to a specific header keyword
    
    ### Parameters:        
    
    input: segyio object.
    
    output: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    ### Examples:

    >>> view.difference(data,data_filt, key = "src", index = 51)
    >>> view.difference(data,data_filt, key = "rec", index = 203)
    >>> view.difference(data,data_filt, key = "cmp", index = 315)
    >>> view.difference(data,data_filt, key = "off", index = 223750)
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(input, key)[0] 

    mng.__check_keyword(key)

    byte = mng.__keywords.get(key)

    traces = np.where(input.attributes(byte)[:] == index)[0]
    
    nt = input.attributes(115)[0][0]
    dt = input.attributes(117)[0][0] * 1e-6

    seismic_input = input.trace.raw[:].T
    seismic_input = seismic_input[:, traces]

    seismic_output = output.trace.raw[:].T
    seismic_output = seismic_output[:, traces]

    seismic_diff = seismic_output - seismic_input

    scale = 0.9*np.std(seismic_output)

    fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (18, 5))

    def set_config(p, fx):
                
        xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
        xlab = traces[xloc]
    
        tloc = np.linspace(0, nt-1, 11, dtype = int)
        tlab = np.around(tloc*dt, decimals = 3)
        
        ax[p].set_yticks(tloc)
        ax[p].set_yticklabels(tlab)
        ax[p].set_xticks(xloc)
        ax[p].set_xticklabels(xlab)
        
        ax[p].set_xlabel('Trace number', fontsize = 15)
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

def radon_transform(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot the Radon transform from a cmp gather.
    
    ### Parameters:        
    
    input: segyio object.
        
    index: integer that select a cmp gather.  

    style: Radon transform type -> ["linear", "parabolic", "hyperbolic"]    

    ### Examples:

    '''  

    key = "cmp"

    byte = mng.__keywords.get(key) 

    _, cmps_per_traces = np.unique(data.attributes(byte)[:], return_counts = True)

    complete_cmp_indexes = np.where(cmps_per_traces == np.max(cmps_per_traces))[0]

    index = kwargs.get("index") if "index" in kwargs else complete_cmp_indexes[0] 

    mng.__check_index(data, key, index)

    style = kwargs.get("style").lower() if "style" in kwargs else "parabolic"

    vmin = kwargs.get("vmin") if "vmin" in kwargs else 1500
    vmax = kwargs.get("vmax") if "vmax" in kwargs else 6000
    nvel = kwargs.get("nvel") if "nvel" in kwargs else 101

    traces = np.where(data.attributes(byte)[:] == index)[0]

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    offset = data.attributes(37)[traces] / data.attributes(69)[traces]

    times = np.arange(nt) * dt
    curvs = np.linspace(vmin, vmax, nvel)

    radon = np.zeros((nt, nvel))

    for i in range(nt):
        for j in range(nvel):

            if style == "linear":
                curvature = times[i] + np.abs(offset)/curvs[j]        
            
            elif style == "parabolic":
                curvature = times[i] + (offset/curvs[j])**2
            
            elif style == "hyperbolic":
                curvature = np.sqrt(times[i]**2 + (offset/curvs[j])**2)            
            
            else:
                print(f"Error: \033[31m{style}\033[m style is not defined!")
                print("Use a valid style: ['linear', 'parabolic', 'hyperbolic']")
                exit()    
        
            mask = np.logical_and(curvature >= 0, curvature <= (nt-1)*dt)

            x = np.arange(len(traces))[mask]
            t = np.array(curvature[mask]/dt, dtype = int)
    
            radon[i,j] = np.sum(seismic[t,x])             


    scale1 = 2*np.std(seismic)
    scale2 = 2*np.std(radon)

    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 5)) 

    ax[0].imshow(seismic, aspect = 'auto', cmap = 'Greys', vmin = -scale1, vmax = scale1) 

    ax[1].imshow(radon, aspect = 'auto', cmap = 'jet', vmin = -scale2, vmax = scale2)

    # ax.set_xlabel('Time [s]', fontsize=15) 
    # ax.set_ylabel('Velocity [m/s]', fontsize=15)

    # cbar = fig.colorbar(img, ax=ax) 
    # cbar.set_label("Semblance", fontsize=15)

    fig.tight_layout() 
    plt.show()




### TESTE DA FUNÇAO -----

def radon_transform2(data : sgy.SegyFile, key : str, index : int, style : str) -> None:
    # Jonatas CMP domain
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # plot: data | radon transform (no wiggle)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    style = ["linear", "parabolic", "hyperbolic"]

    @jit(nopython=True)

    def __radon_adjoint(d,Nt,dt,Nh,h,Np,p,href):

    # Adjoint Time-domain Parabolic Radon Operator

    # d(nt,nh): data                             ##### seismic
    # dt      : sampling interval                ##### dt = data.attributes(117)[0][0] * 1e-6
    # h(Nh)   : offset                           ##### offset = data.attributes(37)[:]
    # p(Np)   : curvature of parabola
    # href    : reference offset
    # Returns m(nt,np) the Radon coefficients 

    # M D Sacchi, 2015,  Email: msacchi@ualberta.ca

    
        m=np.zeros((Nt,Np))

        for itau in range(0,Nt):
            for ih in range(0,Nh):
                for ip in range(0,Np):
                    t = (itau)*dt + p[ip]*(h[ih]/href)**2
                    it = int(t/dt)
                    if it<Nt:
                        if it>0:
                            m[itau,ip] +=  d[it,ih]
        
        return m
            
    @jit(nopython=True)
    def __radon_forward(m,Nt,dt,Nh,h,Np,p,href):

    # Forward Time-domain Parabolic Radon Transform

    # m(nt,nh): Radon coefficients 
    # dt      : sampling interval                                            ##### dt = data.attributes(117)[0][0] * 1e-6
    # h(Nh)   : offset                                                       ##### offset = data.attributes(37)[:]
    # p(Np)   : curvature of parabola
    # href    : reference offset
    # Returns d(nt,nh) the synthetized data from the Radon coefficients

    # M D Sacchi, 2015,  Email: msacchi@ualberta.ca

        d=np.zeros((Nt,Nh))
        
        for itau in range(0,Nt):
            for ih in range(0,Nh):
                for ip in range(0,Np):
                    t = (itau)*dt+p[ip]*(h[ih]/href)**2
                    it=int(t/dt)
                    if it<Nt:
                        if it>=0:
                            d[it,ih] +=  m[itau,ip]                   
        return d
        

    def __radon_cg(d,m0,Nt,dt,Nh,h,Np,p,href,Niter):
            
    # LS Radon transform. Finds the Radon coefficients by minimizing
    # ||L m - d||_2^2 where L is the forward Parabolic Radon Operator.
    # The solution is found via CGLS with operators L and L^T applied on the
    # flight

    # M D Sacchi, 2015,  Email: msacchi@ualberta.ca

        m = m0  
        
        s = d-__radon_forward(m,Nt,dt,Nh,h,Np,p,href) # d - Lm
        pp = __radon_adjoint(s,Nt,dt,Nh,h,Np,p,href)  # pp = L's 
        r = pp
        q = __radon_forward(pp,Nt,dt,Nh,h,Np,p,href)
        old = np.sum(np.sum(r*r))
        print("iter","  res")
        
        for k in range(0,Niter):
             alpha = np.sum(np.sum(r*r))/np.sum(np.sum(q*q))
             m +=  alpha*pp
             s -=  alpha*q
             r = __radon_adjoint(s,Nt,dt,Nh,h,Np,p,href)  # r= L's
             new = np.sum(np.sum(r*r))
             print(k, new)
             beta = new/old
             old = new
             pp = r + beta*pp
             q = __radon_forward(pp,Nt,dt,Nh,h,Np,p,href) # q=L pp
            
        return m
    
#     def __ricker(dt,f0):    
         
# #Ricker wavelet of central frequency f0 sampled every dt seconds

# # M D Sacchi, 2015,  Email: msacchi@ualberta.ca

#         nw = 2.5/f0/dt
#         nw = 2*int(nw/2)
#         nc = int(nw/2)
#         a = f0*dt*3.14159265359
#         n = a*np.arange(-nc,nc)
#         b = n**2
#         return (1-2*b)*np.exp(-b)

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte = mng.__keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]
    offset = data.attributes(37)[traces] / data.attributes(69)[traces]


    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]
    
    # sx = data.attributes(73)[traces] / data.attributes(71)[traces]
    # sy = data.attributes(77)[traces] / data.attributes(71)[traces]    
    # sz = data.attributes(45)[traces] / data.attributes(71)[traces]

    # rx = data.attributes(81)[traces] / data.attributes(69)[traces]
    # ry = data.attributes(85)[traces] / data.attributes(69)[traces]    
    # rz = data.attributes(41)[traces] / data.attributes(69)[traces]
    
    # distance = np.sqrt((sx - rx)**2 + (sy - ry)**2 + (sz - rz)**2)
    times = np.arange(nt) * dt
    nx = len(traces)

    h = offset  # offset
    
    # dh = np.median(np.abs(distance[1:] - distance[:-1])) 
    dh = 25
    
    Nh = len(offset)
    #Np = 55        # Curvatures
    Np = 101
    # curvs = np.linspace(-0.1,.2,Np)
    curvs = np.linspace(1000,3000,Np)
    for i in range(nt):
        for j in range(Np):
            p = np.sqrt(times[i]**2 + (offset/curvs[j])**2) 
    
    #h = np.linspace(0,(Nh-1)*dh,Nh)

    m = np.zeros((nt,Np))

    m0 = np.zeros((nt,Np))
    #f0 = 14

    # wavelet = __ricker(dt,f0)

    href = h[Nh-1]

    # m[40:40+Nw,20]=wavelet
    # m[90:90+Nw,24]=-wavelet
    # m[95:95+Nw,14]=-wavelet
    # m[15:15+Nw,4]=wavelet

    # m[75:75+Nw,12]=-wavelet

    m = __radon_cg(seismic,m0,nt,dt,Nh,h,Np,p,href,10)  # Compute m via inversion using Conjugate Gradients 
    dp = __radon_forward(m,nt,dt,Nh,h,Np,p,href)  # Predict data from inverted m
    
    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]
    print('traces = ', traces)
    tloc = np.linspace(0, nt-1, 11, dtype = int)
    tlab = np.around(tloc*dt, decimals = 1)
    
    scale = 0.8*np.std(seismic)
    
    fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (20, 6))

    iseismic = ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
    ax[0].set_yticks(tloc)
    ax[0].set_yticklabels(tlab)
    ax[0].set_xticks(xloc)
    ax[0].set_xticklabels(xlab)

    ax[0].set_ylabel('Time [s]', fontsize = 15)
    ax[0].set_xlabel('Trace number', fontsize = 15)
    ax[0].cbar = fig.colorbar(iseismic, ax = ax[0])
    ax[0].cbar.set_label("Amplitude", fontsize = 15)

    icurvature = ax[1].imshow(m, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
    ax[1].set_yticks(tloc)
    ax[1].set_yticklabels(tlab)    
    ax[1].set_title('Radon')
    ax[1].set_xlabel('Curvature [s]', fontsize = 15)
    ax[1].set_ylabel("Time [s]", fontsize = 15)
    ax[1].cbar = fig.colorbar(icurvature, ax = ax[1])
    ax[1].cbar.set_label("Amplitude", fontsize = 15)
    
    idp = ax[2].imshow(dp, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
    ax[2].set_yticks(tloc)
    ax[2].set_yticklabels(tlab)    
    ax[2].set_title('Predicted data')
    ax[2].set_xlabel('Offset [m]', fontsize = 15)
    ax[2].set_ylabel("Time [s]", fontsize = 15)
    ax[2].cbar = fig.colorbar(idp, ax = ax[2])
    ax[2].cbar.set_label("Amplitude", fontsize = 15)
    
    fig.tight_layout()
    plt.show()

    pass
    

def semblance(data : sgy.SegyFile, **kwargs):
    """
    Plot the velocity semblance of the according CMP Gather

    ### Mandatory Parameters:

    data: segyio object.

    ### Other Parameters

    index: CMP gather index. - First Complete CMP as Default
    
    vmin: minimum velocity in semblance. - 500.0 ms as Default
    
    vmax: maximum velocity in semblance. - 5000.0 ms as Default

    dv: velocity variation in semblance. - 250.0 ms as Default


    ### Examples:
    
    >>> view.semblance(data)
    >>> view.semblance(data, index=index, vmax=vmax)
    
    """

    key = 'cmp'

    byte = mng.__keywords.get(key) 

    _, cmps_per_traces = np.unique(data.attributes(byte)[:], return_counts = True)

    complete_cmp_indexes = np.where(cmps_per_traces == np.max(cmps_per_traces))[0]

    index = kwargs.get("index") if "index" in kwargs else complete_cmp_indexes[0] 

    mng.__check_index(data, key, index)

    vmin = kwargs.get("vmin") if "vmin" in kwargs else 500.0
    vmax = kwargs.get("vmax") if "vmax" in kwargs else 5000.0 
    dv = kwargs.get("dv") if "dv" in kwargs else 250.0

    traces = np.where(data.attributes(byte)[:] == index)[0]  
    
    seismic = data.trace.raw[:].T  
    seismic = seismic[:, traces]  

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    times = np.arange(nt) * dt
    velocities = np.arange(vmin, vmax+dv, dv)  
    
    nv = len(velocities)   

    offset = data.attributes(37)[traces] / data.attributes(69)[traces]

    semblance = np.zeros((nt, nv))
    points = []
    
    
    for i in range(nt): 
        for j in range(nv):  
            
            moveout = np.array(np.sqrt(times[i]**2 + (offset/velocities[j])**2) / dt, dtype = int)   

            mask = moveout < nt

            t = moveout[mask]
            x = np.arange(len(traces))[mask]
    
            semblance[i, j] += np.sum(np.abs(seismic[t, x]))**2         
    def onclick(event):

        
        global ponto_parada
        
        if event.inaxes is not None:
        
            x, y = event.xdata, event.ydata
            if ponto_parada is None:
                points.append((x, y))
                plt.plot(x, y, 'ro')
                plt.draw()
                print(points)
            else:
                points.append((x, y))
                plt.plot(x, y, 'bo')
                print(points)
            plt.draw()
            
            # for i in range (1,camadas):
            #     vint[i]=np.sqrt(((vrms[i])**2*points[i]-(vrms[i-1])**2*points[1])/(tt[i]-tt[i-1]))
        else:
            print('Clique fora dos eixos.')
    def on_key(event):
        if event.key=='n' and len(points) >= 1:
            index = int(input('Escolha o ponto para deletar (começando de 0): '))
            
            if 0 <= index < len(points):
                points.pop(index)
                ax[1].clear()
                im2 = ax[1].imshow(semblance, aspect='auto', cmap='jet', extent=[vmin, vmax, times[-1], times[0]])
                ax[1].set_xlabel('Velocity [m/s]', fontsize=15)
                ax[1].set_ylabel('Time [s]', fontsize=15)
                
                ax[1].set_xlim(vmin, vmax)

                ax[1].set_xlabel('Velocity [m/s]', fontsize=15)
                ax[1].set_ylabel('Time [s]', fontsize=15)
    
                
                
    
                for p in points:
                    plt.plot(p[0], p[1], 'ro')
                    plt.draw()
    def stop(event):
         global ponto_parada
         if event.key == 'j':
        
            if ponto_parada is None:  
                ponto_parada = (0, 0)
                
                plt.plot(ponto_parada[0], ponto_parada[1], 'go', label='Ponto de Parada')
                plt.draw()    
    
    def onkeypress(event):
    
        global interpolated_line

        if event.key == 'enter':
            if len(points) > 1:
                points_sorted = sorted(points, key = lambda p: p[1])
                x,y = zip(*points_sorted)
               
                


                cs = CubicSpline(y, x, bc_type='natural')
                ynew = np.linspace(min(y), max(y), num = 500)
                xnew = cs(ynew)
                if interpolated_line:
                    interpolated_line.remove()

                interpolated_line, = plt.plot(xnew, ynew)
                plt.draw()
        elif event.key =='m':
            if interpolated_line:
                interpolated_line.remove()  
                interpolated_line = None  
                plt.draw()
    def save(event):
        if event.key=='c':
            np.savetxt("coordernada", points, fmt = "%.6f")
            np.savetxt("coordernadainter", points, fmt = "%.6f")



    
    
    vmin_gather = np.percentile(seismic, 1)
    vmax_gather = np.percentile(seismic, 99) 

    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 5)) 

    im1 = ax[0].imshow(seismic, aspect='auto', cmap='Greys', vmin=vmin_gather, vmax=vmax_gather, extent=[0, seismic.shape[1], times[-1], times[0]])
    ax[0].set_xlabel('Trace number', fontsize=15)
    ax[0].set_ylabel('Time [s]', fontsize=15)
    cbar1 = fig.colorbar(im1, ax=ax[0])
    cbar1.set_label("Amplitude", fontsize=10)

    im2 = ax[1].imshow(semblance, aspect='auto', cmap='jet', extent=[vmin, vmax, times[-1], times[0]])
    ax[1].set_xlabel('Velocity [m/s]', fontsize=15)
    ax[1].set_ylabel('Time [s]', fontsize=15)
    cbar2 = fig.colorbar(im2, ax=ax[1])
    cbar2.set_label("Semblance", fontsize=10)

    ax[1].set_xlim(vmin, vmax)

    ax[1].set_xlabel('Velocity [m/s]', fontsize=15)
    ax[1].set_ylabel('Time [s]', fontsize=15)
    
    fig.suptitle(f'Velocity Semblance - Curent CMP: {index}', fontsize=16)

    fig.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkeypress)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('key_press_event', stop)
    fig.canvas.mpl_connect('key_press_event', save) 
    plt.show()
    

