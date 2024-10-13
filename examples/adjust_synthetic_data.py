import sys; sys.path.append("../")

import numpy, segyio 

import matplotlib.pyplot as plt

from toolbox import managing as mng

folder = "../data/synthetic/"

nt = 6001
dt = 1e-3

ns = 477
nr = 877 

dr = 25.0
ds = 25.0

scalar = 100
spread = 321

SPS = numpy.zeros((ns, 2))
RPS = numpy.zeros((nr, 2)) 
XPS = numpy.zeros((ns, 2))

SPS[:,0] = numpy.arange(0, 11900 + ds, ds)
RPS[:,0] = numpy.arange(100, 22000 + dr, dr)

SPS[:,1] = 25.0
RPS[:,1] = 25.0

XPS[:,0] = numpy.arange(ns)  
XPS[:,1] = numpy.arange(ns) + spread 

data = numpy.zeros((nt, ns*spread))

tsl = numpy.zeros(ns * spread, dtype = int)
tsf = numpy.zeros(ns * spread, dtype = int)

tsi = numpy.zeros(ns * spread, dtype = int) + int(dt*1e6)
tsc = numpy.zeros(ns * spread, dtype = int) + nt

src = numpy.zeros(ns * spread, dtype = int)
rec = numpy.zeros(ns * spread, dtype = int)

off = numpy.zeros(ns * spread, dtype = int)
cmp = numpy.zeros(ns * spread, dtype = int)
cmpx = numpy.zeros(ns * spread, dtype = int)
cmpy = numpy.zeros(ns * spread, dtype = int)

xsrc = numpy.zeros(ns * spread, dtype = int)
ysrc = numpy.zeros(ns * spread, dtype = int)
zsrc = numpy.zeros(ns * spread, dtype = int)

xrec = numpy.zeros(ns * spread, dtype = int)
yrec = numpy.zeros(ns * spread, dtype = int)
zrec = numpy.zeros(ns * spread, dtype = int)

gscal = numpy.zeros(ns * spread, dtype = int) + scalar

spread_index = numpy.arange(spread)

for shot in range(ns):
    
    seismic = numpy.fromfile(folder + f"elastic_iso_data_nStations{spread}_nSamples{nt}_shot_{shot+1}.bin", dtype = numpy.float32, count = nt*spread)
    seismic = numpy.reshape(seismic, [nt, spread], order = "F")

    fill = slice(shot*spread, shot*spread + spread)

    data[:,fill] = seismic[:,:]

    tsl[fill] = 1 + spread_index + shot*spread
    
    tsf[fill] = 1 + XPS[shot,0]
    src[fill] = 1 + XPS[shot,0]
    
    xsrc[fill] = SPS[shot,0]*scalar
    zsrc[fill] = SPS[shot,1]*scalar

    rec[fill] = 1 + numpy.arange(spread) + shot

    xrec[fill] = RPS[rec[fill]-1, 0]*scalar    
    zrec[fill] = RPS[rec[fill]-1, 1]*scalar    

    off[fill] = (xrec[fill] - xsrc[fill])      
    
    cmp[fill] = numpy.arange(spread, dtype = int) + 2.0*(ds/dr)*shot + 1
    
    cmpx[fill] = xsrc[fill] - 0.5*(xsrc[fill] - xrec[fill]) 
    cmpy[fill] = ysrc[fill] - 0.5*(ysrc[fill] - yrec[fill]) 


segyio.tools.from_array2D(folder + f"synthetic_data_raw.sgy", data.T)

data = mng.import_sgy_file(folder + f"synthetic_data_raw.sgy")

bytes = [1, 5, 9, 13, 37, 21, 69, 115, 117, 
         41, 45, 73, 77, 81, 85, 181, 185]

values = [tsl, tsf, src, rec, off, cmp, gscal, 
          tsc, tsi, zrec, zsrc, xsrc, ysrc, 
          xrec, yrec, cmpx, cmpy]

mng.edit_trace_header(data, bytes, values)

mng.show_trace_header(data)
