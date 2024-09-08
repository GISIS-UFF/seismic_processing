import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filt
from toolbox import visualizing as view

import numpy as np
import segyio as sgy

import matplotlib.pyplot as plt

def analytical_reflections(v, z, x):
    Tint = 2.0 * z / v[:-1]
    Vrms = np.zeros(len(z))
    reflections = np.zeros((len(z), len(x)))
    for i in range(len(z)):
        Vrms[i] = np.sqrt(np.sum(v[:i+1]**2 * Tint[:i+1]) / np.sum(Tint[:i+1]))
        reflections[i] = np.sqrt(x**2.0 + 4.0*np.sum(z[:i+1])**2) / Vrms[i]
    return reflections

def wavelet_generation(nt, dt, fmax):
    ti = (nt/2)*dt
    fc = fmax / (3.0 * np.sqrt(np.pi)) 
    wavelet = np.zeros(nt)
    for n in range(nt):            
        arg = np.pi*((n*dt - ti)*fc*np.pi)**2    
        wavelet[n] = (1.0 - 2.0*arg)*np.exp(-arg);      
    return wavelet

def radon_transform(gather, dt, times, offsets, velocities, style, operation):

    Nt = len(times)
    Nh = len(offsets)
    Np = len(velocities)

    domain = np.zeros((Nt, Nh, Np))

    offset_index = np.arange(Nh, dtype = int)

    for tind in range(Nt): 
        for vind in range(Np):  

            if style == "linear":
                curvature = times[tind] + offsets/velocities[vind]        
            
            elif style == "parabolic":
                curvature = times[tind] + (offsets/velocities[vind])**2
            
            elif style == "hyperbolic":
                curvature = np.sqrt(times[tind]**2 + (offsets/velocities[vind])**2)            

            else:
                raise ValueError(f"Error: {style} style is not defined! Use a valid style: ['linear', 'parabolic', 'hyperbolic']")

            mask = curvature <= (Nt-1)*dt

            t = np.array(curvature[mask]/dt, dtype = int)
                        
            domain[tind, mask, vind] = gather[t, offset_index[mask]] / Np
            
    if operation == "forward":
        return np.sum(domain, axis = 1)

    elif operation == "adjoint":
        return np.sum(domain, axis = 2)

    else:
        raise ValueError(f"Error: {operation} operation is not defined! Use a valid operation: ['forward', 'adjoint']")

# def radon_cg(data, nt, dt, Nh, offsets, Np, curvature, href, iterations):
        
# # LS Radon transform. Finds the Radon coefficients by minimizing
# # ||L m - d||_2^2 where L is the forward Parabolic Radon Operator.
# # The solution is found via CGLS with operators L and L^T applied on the
# # flight

# # M D Sacchi, 2015,  Email: msacchi@ualberta.ca
#     #m = np.zeros((nt,Np))

#     m0 = np.zeros((nt,Np))
#     m = m0  
    
#     s = data - __radon_operator('forward', m, nt, dt, Nh, offsets, Np, curvature, href) # d - Lm
#     pp = __radon_operator('adjoint', s, nt, dt, Nh, offsets, Np, curvature, href)  # pp = L's 
#     r = pp
#     q = __radon_operator('forward',pp, nt, dt, Nh, offsets, Np, curvature, href)
#     old = np.sum(np.sum(r*r))
#     print("iter","  res")
    
#     for k in range(0,iterations):
#             alpha = np.sum(np.sum(r*r))/np.sum(np.sum(q*q))
#             m +=  alpha*pp
#             s -=  alpha*q
#             r = __radon_operator('adjoint',s, nt, dt, Nh, offsets, Np, curvature, href)  # r= L's
#             new = np.sum(np.sum(r*r))
#             print(k, new)
#             beta = new/old
#             old = new
#             pp = r + beta*pp
#             q = __radon_operator('forward', pp , nt, dt, Nh, offsets, Np, curvature, href) # q=L pp
        
#     return m


n_receivers = 320
spread_length = 8000
total_time = 5.0
fmax = 30.0

dx = 25
dt = 1e-3

nt = int(total_time / dt) + 1
nx = int(n_receivers / 2) + 1

z = np.array([500, 1000, 1000, 1000])
v = np.array([1500, 1650, 2000, 3000, 4500])

x = np.linspace(0, nx*dx, nx)

reflections = analytical_reflections(v, z, x)

seismogram = np.zeros((nt, nx), dtype = np.float32)
wavelet = wavelet_generation(nt, dt, fmax)

for j in range(nx):
    for i in range(len(z)):
        indt = int(reflections[i, j] / dt)
        seismogram[indt, j] = 1.0

    seismogram[:,j] = np.convolve(seismogram[:, j], wavelet, "same")

Nt = 5001
Nh = 160
Np = 101

dt = 1e-3

x = np.linspace(0, nx*dx, nx)
v = np.linspace(1000, 3000, Np) 
t = np.arange(Nt) * dt

m = radon_transform(seismogram, dt, t, x, v, "hyperbolic", "forward")
d = radon_transform(seismogram, dt, t, x, v, "hyperbolic", "adjoint")

plt.subplot(131)
plt.imshow(seismogram, aspect = "auto")

plt.subplot(132)
plt.imshow(m, aspect = "auto")

plt.subplot(133)
plt.imshow(d, aspect = "auto")

plt.show()
