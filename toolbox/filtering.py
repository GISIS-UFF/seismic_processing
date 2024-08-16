import scipy as sc
import numpy as np
import segyio as sgy
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path

from numba import jit

from toolbox import managing as mng

def fourier_fx_domain(input : sgy.SegyFile, fmin : float, fmax : float, output_name : str) -> sgy.SegyFile:

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

def fourier_fk_domain(data: sgy.SegyFile, angle: float) -> sgy.SegyFile:
    current_clicks = []
    polygons = []
    polygon_paths = []

    def on_click(event):
        global current_clicks
        if event.inaxes:
            current_clicks.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')
            plt.draw()

            if len(current_clicks) >= 3:
                for artist in ax.patches:
                    artist.remove()

                for polygon in polygons:
                    ax.add_patch(polygon)

                polygon = Polygon(current_clicks, closed=True, edgecolor='black', facecolor='cyan', alpha=0.5)
                ax.add_patch(polygon)
                plt.draw()

    def on_key(event):
        global current_clicks, polygons, polygon_paths
        if event.key == 'n':
            if len(current_clicks) >= 3:
                polygon = Polygon(current_clicks, closed=True, edgecolor='black', facecolor='cyan', alpha=0.5)
                polygons.append(polygon)
                polygon_path = Path(current_clicks)
                polygon_paths.append(polygon_path)

            current_clicks = []

    fig, ax = plt.subplots()
    ax.imshow(data, cmap='gray')
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if polygon_paths:
        mask = np.zeros(data.shape)

        x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        xy = np.vstack((x.flatten(), y.flatten())).T

        for polygon_path in polygon_paths:
            inside = polygon_path.contains_points(xy).reshape(mask.shape)
            mask[inside] = 1

        mask = gaussian_filter(mask.astype(np.float32), sigma=7)

        masked_image = data * mask

        plt.figure()
        plt.imshow(masked_image, cmap='gray')
        plt.show()

def apply_mute():
    pass

### TESTE DA FUNÇAO -----

def radon_transform2(data : sgy.SegyFile, key : str, index : int, style : str) -> None:
    # Jonatas CMP domain
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # plot: data | radon transform (no wiggle)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    style = ["linear", "parabolic", "hyperbolic"]

    @jit(nopython=True)

    def __radon_adjoint(d,Nt,dt,Nh,offset,Np,curvature,href):

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
                    t =np.sqrt((itau*dt)**2 + (curvature[ip]* (offset[ih]/href))**2)
                    it = int(t/dt)
                    if it<Nt:
                        if it>0:
                            m[itau,ip] +=  d[it,ih]
        
        return m
            
    @jit(nopython=True)
    def __radon_forward(m, Nt,dt,Nh,offset,Np,curvature,href):

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
                    t =np.sqrt((itau*dt)**2 + (curvature[ip]* (offset[ih]/href))**2)
                    it=int(t/dt)
                    if it<Nt:
                        if it>=0:
                            d[it,ih] +=  m[itau,ip]                   
        return d
        

    def __radon_cg(d,nt,dt,Nh,offset,Np,curvature,href,Niter):
            
    # LS Radon transform. Finds the Radon coefficients by minimizing
    # ||L m - d||_2^2 where L is the forward Parabolic Radon Operator.
    # The solution is found via CGLS with operators L and L^T applied on the
    # flight

    # M D Sacchi, 2015,  Email: msacchi@ualberta.ca
        #m = np.zeros((nt,Np))

        m0 = np.zeros((nt,Np))
        m = m0  
        
        s = d-__radon_forward(m,nt,dt,Nh,offset,Np,curvature,href) # d - Lm
        pp = __radon_adjoint(s,nt,dt,Nh,offset,Np,curvature,href)  # pp = L's 
        r = pp
        q = __radon_forward(pp,nt,dt,Nh,offset,Np,curvature,href)
        old = np.sum(np.sum(r*r))
        print("iter","  res")
        
        for k in range(0,Niter):
             alpha = np.sum(np.sum(r*r))/np.sum(np.sum(q*q))
             m +=  alpha*pp
             s -=  alpha*q
             r = __radon_adjoint(s,nt,dt,Nh,offset,Np,curvature,href)  # r= L's
             new = np.sum(np.sum(r*r))
             print(k, new)
             beta = new/old
             old = new
             pp = r + beta*pp
             q = __radon_forward(pp,nt,dt,Nh,offset,Np,curvature,href) # q=L pp
            
        return m
    

    mng.__check_keyword(key)
    # mng.__check_index(data, key, index)

    byte = mng.__keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]
    offset = data.attributes(37)[:] / data.attributes(69)[:]
    print('offset = ', offset)

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]
    
    times = np.arange(nt) * dt
    
    Nh = len(offset)
    Np = 55        # Curvatures

    curvs = np.linspace(1000, 3000,Np)
    for i in range(nt):
        for j in range(Np):
            curvature = np.sqrt(times[i]**2 + (offset/curvs[j])**2) 

    m = np.zeros((nt,Np))

    # href = np.max(offset)
    href = 8000

    m = __radon_cg(seismic,nt,dt,Nh,offset,Np,curvature,href,10)  # Compute m via inversion using Conjugate Gradients 
    dp = __radon_forward(m, nt,dt,Nh,offset,Np,curvature,href)  # Predict data from inverted m
    
    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]
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