import sys; sys.path.append("../")

import numpy as np
import numba as nb

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

def radon_forward_s(data, dt, times, offsets, velocities, style):

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

            domain[tind, mask, vind] = data[t, offset_index[mask]]

    return domain

@nb.jit(nopython = True, parallel = True)
def radon_forward(data, dt, times, offsets, velocities, style):

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
                        domain[tind, hind, vind] = data[int(curvature/dt), hind]           

    elif style == "parabolic":
        for tind in nb.prange(Nt): 
            for vind in nb.prange(Np):  
                for hind in nb.prange(Nh):
                    curvature = times[tind] + (offsets[hind]/velocities[vind])**2
                    if 0 <= curvature <= (Nt-1)*dt:
                        domain[tind, hind, vind] = data[int(curvature/dt), hind]           
                
    elif style == "hyperbolic":
        for tind in nb.prange(Nt): 
            for vind in nb.prange(Np):  
                for hind in nb.prange(Nh):
                    curvature = np.sqrt(times[tind]**2 + (offsets[hind]/velocities[vind])**2)            
                    if 0 <= curvature <= (Nt-1)*dt:
                        domain[tind, hind, vind] = data[int(curvature/dt), hind]           

    else:
        raise ValueError(f"Error: {style} style is not defined! Use a valid style: ['linear', 'parabolic', 'hyperbolic']")

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
    
    else:
        raise ValueError(f"Error: {style} style is not defined! Use a valid style: ['linear', 'parabolic', 'hyperbolic']")

    return data




n_receivers = 320
spread_length = 8000
total_time = 5.0
fmax = 30.0

dx = 25
dt = 1e-3

Nt = int(total_time / dt) + 1
Nh = int(n_receivers / 2) + 1

z = np.array([500, 1000, 1000, 1000])
v = np.array([1500, 1650, 2000, 3000, 4500])

offsets = np.linspace(0, Nh*dx, Nh)

reflections = analytical_reflections(v, z, offsets)

seismogram = np.zeros((Nt, Nh), dtype = np.float32)
wavelet = wavelet_generation(Nt, dt, fmax)

for j in range(Nh):
    for i in range(len(z)):
        indt = int(reflections[i, j] / dt)
        seismogram[indt, j] = 1.0

    seismogram[:,j] = np.convolve(seismogram[:, j], wavelet, "same")

Np = 101

x = offsets.copy()
v = np.linspace(1000, 3000, Np) 
t = np.arange(Nt) * dt


### Performance test

from time import time

ti = time()
domain = radon_forward_s(seismogram, dt, t, x, v, "hyperbolic")
tf = time()

print(f"radon forward serial: {tf - ti} s")

radon_s = np.sum(domain, axis = 1)

ti = time()
domain = radon_forward(seismogram, dt, t, x, v, "hyperbolic")
tf = time()

print(f"radon forward parallel: {tf - ti} s")

radon_p = np.sum(domain, axis = 1)

### Accuracy test

domain = radon_forward(seismogram, dt, t, x, v, "hyperbolic")
data = radon_adjoint(domain, dt, t, x, v, "hyperbolic")

difference = (seismogram - data)**2

print(np.sum(difference))

plt.subplot(131)
plt.imshow(seismogram, aspect = "auto")

plt.subplot(132)
plt.imshow(radon_s, aspect = "auto")

plt.subplot(133)
plt.imshow(radon_p, aspect = "auto")

plt.show()


plt.subplot(131)
plt.imshow(seismogram, aspect = "auto")

plt.subplot(132)
plt.imshow(radon_p, aspect = "auto")

plt.subplot(133)
plt.imshow(difference, aspect = "auto")

plt.show()
