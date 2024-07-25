import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

from toolbox import managing as mng

def gather(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot a prestack seismic gather according to a specific header keyword.

    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    ### Hint:

    If no keys or indexes provided, it will be plot the first shot gather.  
        
    ### Examples:

    >>> view.gather(data)
    >>> view.gather(data, key = "src", index = 51)
    >>> view.gather(data, key = "rec", index = 203)
    >>> view.gather(data, key = "cmp", index = 315)
    >>> view.gather(data, key = "off", index = 223750)
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte, label = mng.__keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    scale = 0.8*np.std(seismic)

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

def geometry(data : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot geometry, traces per cmp and the current source-receiver configuration 
    according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    ### Hint:

    If no keys or indexes provided, it will be plot the first shot gather.  

    ### Examples:

    >>> view.geometry(data, key = "src", index = 51)
    >>> view.geometry(data, key = "rec", index = 203)
    >>> view.geometry(data, key = "cmp", index = 315)
    >>> view.geometry(data, key = "off", index = 223750)
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte, label = mng.__keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    sx_complete = data.attributes(73)[:]  
    sy_complete = data.attributes(77)[:]    

    rx_complete = data.attributes(81)[:]
    ry_complete = data.attributes(85)[:]    

    cmpx = data.attributes(181)[:] 
    cmpy = data.attributes(185)[:] 

    fig, ax = plt.subplots(num = f"Common {label} gather geometry", ncols = 3, nrows = 1, figsize = (15, 5))

    def set_config(p):
        ax[p].set_xlabel("X [m]", fontsize = 15)
        ax[p].set_ylabel("y [m]", fontsize = 15)
        ax[p].legend(loc = "lower left")

    ax[0].plot(rx_complete, ry_complete, "v", label = "Receivers")
    ax[0].plot(sx_complete, sy_complete, "*", label = "Sources")
    ax[0].set_title("Complete geometry", fontsize = 15)
    set_config(0)

    ax[1].plot(cmpx, cmpy, ".", label = "CMP")
    ax[1].set_title("Coverage", fontsize = 15)
    set_config(1)

    ax[2].plot(rx_complete[traces], ry_complete[traces], "v", label="Receivers")
    ax[2].plot(sx_complete[traces], sy_complete[traces], "*", label="Sources")
    ax[2].set_title("Local geometry", fontsize = 15)
    set_config(2)

    fig.tight_layout()
    plt.show()

def fourier_fx_domain(data : sgy.SegyFile, fmin : float, fmax = float, **kwargs) -> None:
    '''
    Plot the amplitude spectra of each trace in gather according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    fmin: minimum frequency to visualize    
    
    fmin: maximum frequency to visualize    

    ### Hint:

    If no keys or indexes provided, it will be plot the first shot gather.  
    
    ### Examples:

    >>> view.fourier_fx_domain(data, key = "src", index = 51, fmin = 0, fmax = 100)
    >>> view.fourier_fx_domain(data, key = "rec", index = 203, fmin = 0, fmax = 100)
    >>> view.fourier_fx_domain(data, key = "cmp", index = 315, fmin = 0, fmax = 100)
    >>> view.fourier_fx_domain(data, key = "off", index = 223750, fmin = 0, fmax = 100)
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte, label = mng.__keywords.get(key)

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
    flab = np.around(np.ceil(frequency[floc]), decimals = 1)
    
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

    ax[0].cbar = fig.colorbar(im, ax = ax[0])
    ax[0].cbar.set_label("Amplitude", fontsize = 15) 

    fx = ax[1].imshow(np.abs(fx_seismic[mask,:]), aspect = "auto", cmap = "jet")
   
    ax[1].set_yticks(floc)
    ax[1].set_yticklabels(flab)
    ax[1].set_xticks(xloc)
    ax[1].set_xticklabels(xlab)

    ax[1].set_ylabel("Frequency [Hz]", fontsize = 15)
    ax[1].set_xlabel("Trace number", fontsize = 15)

    ax[1].cbar = fig.colorbar(fx, ax = ax[1])
    ax[1].cbar.set_label("Amplitude", fontsize = 15) 

    fig.tight_layout()
    plt.show()

def fourier_fk_domain(data : sgy.SegyFile, fmin : float, fmax = float, **kwargs) -> None:
    '''
    Plot the amplitude spectra of each trace in gather according to a specific header keyword.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    fmin: minimum frequency to visualize    
    
    fmin: maximum frequency to visualize    

    ### Hint:

    If no keys or indexes provided, it will be plot the first shot gather.  
    
    ### Examples:

    >>> view.fourier_fk_domain(data, fmin = 0, fmax = 100, key = "src", index = 51)
    >>> view.fourier_fk_domain(data, fmin = 0, fmax = 100, key = "rec", index = 203)
    >>> view.fourier_fk_domain(data, fmin = 0, fmax = 100, key = "cmp", index = 315)
    >>> view.fourier_fk_domain(data, fmin = 0, fmax = 100, key = "off", index = 223750)
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte, label = mng.__keywords.get(key)

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
    flab = np.around(np.ceil(frequency[mask][floc][::-1]), decimals = 1)

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

def difference(input : sgy.SegyFile, output : sgy.SegyFile, **kwargs) -> None:
    '''
    Plot a difference between prestack seismic gathers according to a specific header keyword
    
    ### Parameters:        
    
    input: segyio object.
    
    output: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    ### Hint:

    If no keys or indexes provided, it will be plot the first shot gather.  

    ### Examples:

    >>> view.difference(data,data_filt, key = "src", index = 51)
    >>> view.difference(data,data_filt, key = "rec", index = 203)
    >>> view.difference(data,data_filt, key = "cmp", index = 315)
    >>> view.difference(data,data_filt, key = "off", index = 223750)
    '''    

    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(input, key)[0] 

    mng.__check_keyword(key)

    byte, label = mng.__keywords.get(key)

    traces = np.where(input.attributes(byte)[:] == index)[0]
    
    nt = input.attributes(115)[0][0]
    dt = input.attributes(117)[0][0] * 1e-6

    seismic_input = input.trace.raw[:].T
    seismic_input = seismic_input[:, traces]

    seismic_output = output.trace.raw[:].T
    seismic_output = seismic_output[:, traces]

    seismic_diff = seismic_output - seismic_input

    scale = 0.9*np.std(seismic_output)

    fig, ax = plt.subplots(num = f"Common {label} gather", ncols = 3, nrows = 1, figsize = (18, 5))

    def set_config(p, fx):
                
        xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
        xlab = traces[xloc]
    
        tloc = np.linspace(0, nt-1, 11, dtype = int)
        tlab = np.around(tloc*dt, decimals = 1)
        
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

def radon_transform(style : str):
    # Jonatas CMP domain
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # plot: data | radon transform (no wiggle)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    style = ["linear", "parabolic", "hyperbolic"]

    # from numba import jit
    # import numpy as np

    # @jit(nopython=True)
    # def radon_adjoint(d,Nt,dt,Nh,h,Np,p,href):

    # # Adjoint Time-domain Parabolic Radon Operator

    # # d(nt,nh): data
    # # dt      : sampling interval
    # # h(Nh)   : offset
    # # p(Np)   : curvature of parabola
    # # href    : reference offset
    # # Returns m(nt,np) the Radon coefficients 

    # # M D Sacchi, 2015,  Email: msacchi@ualberta.ca

    
    #     m=np.zeros((Nt,Np))

    #     for itau in range(0,Nt):
    #         for ih in range(0,Nh):
    #             for ip in range(0,Np):
    #                 t = (itau)*dt + p[ip]*(h[ih]/href)**2
    #                 it = int(t/dt)
    #                 if it<Nt:
    #                     if it>0:
    #                         m[itau,ip] +=  d[it,ih]
        
    #     return m
            

    # def radon_forward(m,Nt,dt,Nh,h,Np,p,href):

    # # Forward Time-domain Parabolic Radon Transform

    # # m(nt,nh): Radon coefficients 
    # # dt      : sampling interval
    # # h(Nh)   : offset
    # # p(Np)   : curvature of parabola
    # # href    : reference offset
    # # Returns d(nt,nh) the synthetized data from the Radon coefficients

    # # M D Sacchi, 2015,  Email: msacchi@ualberta.ca

    #     d=np.zeros((Nt,Nh))
        
    #     for itau in range(0,Nt):
    #         for ih in range(0,Nh):
    #             for ip in range(0,Np):
    #                 t = (itau)*dt+p[ip]*(h[ih]/href)**2
    #                 it=int(t/dt)
    #                 if it<Nt:
    #                     if it>=0:
    #                         d[it,ih] +=  m[itau,ip]                   
    #     return d
        

    # def radon_cg(d,m0,Nt,dt,Nh,h,Np,p,href,Niter):
            
    # # LS Radon transform. Finds the Radon coefficients by minimizing
    # # ||L m - d||_2^2 where L is the forward Parabolic Radon Operator.
    # # The solution is found via CGLS with operators L and L^T applied on the
    # # flight

    # # M D Sacchi, 2015,  Email: msacchi@ualberta.ca

    #     m = m0  
        
    #     s = d-radon_forward(m,Nt,dt,Nh,h,Np,p,href) # d - Lm
    #     pp = radon_adjoint(s,Nt,dt,Nh,h,Np,p,href)  # pp = L's 
    #     r = pp
    #     q = radon_forward(pp,Nt,dt,Nh,h,Np,p,href)
    #     old = np.sum(np.sum(r*r))
    #     print("iter","  res")
        
    #     for k in range(0,Niter):
    #          alpha = np.sum(np.sum(r*r))/np.sum(np.sum(q*q))
    #          m +=  alpha*pp
    #          s -=  alpha*q
    #          r = radon_adjoint(s,Nt,dt,Nh,h,Np,p,href)  # r= L's
    #          new = np.sum(np.sum(r*r))
    #          print(k, new)
    #          beta = new/old
    #          old = new
    #          pp = r + beta*pp
    #          q = radon_forward(pp,Nt,dt,Nh,h,Np,p,href) # q=L pp
            
    #     return m
    pass

def semblance():
    # Amanda CMP domain

    # nt = 5001
    # dt = 1e-3

    # nx = 161
    # dx = 25.0

    # vi = 1000
    # vf = 3000
    # dv = 50

    # filename = f"cmp_gather_{nt}x{nx}_{dt*1e6:.0f}us.bin"

    # seismic = np.fromfile(filename, dtype = np.float32, count = nt*nx)
    # seismic = np.reshape(seismic, [nt,nx], order = "F")

    # vrms = np.arange(vi, vf + dv, dv)
    # offset = np.arange(nx, dtype = int)

    # time = np.arange(nt) * dt

    # semblance = np.zeros((nt, len(vrms)))

    # for indt, t0 in enumerate(time):
    #     for indv, v in enumerate(vrms):
        
    #         target = np.array(np.sqrt(t0**2 + (offset*dx/v)**2) / dt, dtype = int) 

    #         mask = target < nt
        
    #         t = target[mask]
    #         x = offset[mask]
        
    #         semblance[indt, indv] = np.sum(np.abs(seismic[t,x]))**2    

    # xloc = np.linspace(0, len(vrms)-1, 9)
    # xlab = np.linspace(vi, vf, 9)

    # tloc = np.linspace(0, nt, 11)
    # tlab = np.around(np.linspace(0, nt-1, 11)*dt, decimals = 3)

    # scale = 15.0*np.std(semblance)

    # fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10,8))

    # ax[0].imshow(seismic, aspect = "auto", cmap = "Greys")
    # ax[0].set_yticks(tloc)
    # ax[0].set_yticklabels(tlab)

    # ax[0].set_xticks(np.linspace(0,nx,5))
    # ax[0].set_xticklabels(np.linspace(0,nx-1,5, dtype = int)*dx)

    # ax[0].set_title("CMP Gather", fontsize = 18)
    # ax[0].set_xlabel("Offset [m]", fontsize = 15)
    # ax[0].set_ylabel("Two Way Time [s]", fontsize = 15)

    # ax[1].imshow(semblance, aspect = "auto", cmap = "jet", vmin = -scale, vmax = scale)

    # ax[1].set_xticks(xloc)
    # ax[1].set_xticklabels(xlab*1e-3)

    # ax[1].set_yticks(tloc)
    # ax[1].set_yticklabels(tlab)

    # ax[1].set_title("Semblance", fontsize = 18)
    # ax[1].set_xlabel("RMS Velocity [km/s]", fontsize = 15)
    # ax[1].set_ylabel("Two Way Time [s]", fontsize = 15)

    # fig.tight_layout()
    # plt.grid()
    # plt.show()

    pass

