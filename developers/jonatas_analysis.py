from sys import path
path.append("../")

import toolbox.visualizing as view
from toolbox import managing as mng
from toolbox import filtering

import numpy as np
import segyio as sgy

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

sgy.tools.from_array2D("../data/seismic_radon_test.sgy", seismogram.T)

data = mng.import_sgy_file("../data/seismic_radon_test.sgy")

scalar = 100

tsl = np.arange(nx) + 1
tsf = np.ones(nx)

tsi = np.zeros(nx) + int(dt*1e6)
tsc = np.zeros(nx) + nt

src = np.zeros(nx) + 1001
rec = np.arange(nx) + 2001

off = np.arange(nx) * dx * scalar

cmp = np.zeros(nx) + 1

xsrc = np.zeros(nx) 
ysrc = np.zeros(nx)
zsrc = np.zeros(nx)

xrec = np.arange(nx) * dx * scalar
yrec = np.arange(nx) * dx * scalar
zrec = np.zeros(nx)

cmpx = xsrc - 0.5*(xsrc - xrec)*scalar 
cmpy = ysrc - 0.5*(ysrc - yrec)*scalar

gscal = np.zeros(n_receivers, dtype = int) + scalar

bytes = [1, 5, 9, 13, 37, 21, 69, 115, 117, 
         41, 45, 73, 77, 81, 85, 181, 185]

values = [tsl, tsf, src, rec, off, cmp, gscal, 
          tsc, tsi, zrec, zsrc, xsrc, ysrc, 
          xrec, yrec, cmpx, cmpy]

mng.edit_trace_header(data, bytes, values)

mng.show_trace_header(data)

view.gather(data)
view.geometry(data)
view.radon_transform(data, style = "hyperbolic", index = 1, vmin = 1000, vmax = 3000)

# data2 = mng.import_sgy_file("../data/overthrust_synthetic_seismic_data.sgy")
data2 = mng.import_sgy_file("../data/seismic_radon_test.sgy")


style = 'hyperbolic'
key = 'cmp'
index = 1
filtering.radon_transform2(data2, key, index, style)
