import scipy as sc
import numpy as np
import segyio as sgy

import matplotlib.pyplot as plt

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

def fourier_fk_domain(data : sgy.SegyFile, angle : float) -> sgy.SegyFile:
    pass

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

