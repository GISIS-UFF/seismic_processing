import sys; sys.path.append("../")

import numpy, segyio 

from toolbox import managing as mng

input_file = "../data/2D_Land_vibro_data_2ms/Line_001"

data = mng.import_sgy_file(f"{input_file}.sgy")

mng.show_trace_header(data)

nt = 1501
dt = 2e-3

dr = 25.0
ds = 50.0

scalar = 100

SPS = numpy.loadtxt(f"{input_file}.SPS", dtype = str, comments = "H26", usecols = [1, 7, 8, 9])
RPS = numpy.loadtxt(f"{input_file}.RPS", dtype = str, comments = "H26", usecols = [1, 7, 8, 9])
XPS = numpy.loadtxt(f"{input_file}.XPS", dtype = str, comments = "H26", usecols = [1, 2, 5, 6])

nShots = len(SPS)
spread = int(XPS[:,3][0][:-1]) - int(XPS[:,2][0]) + 1 

seismic = data.trace.raw[:].T

traces_to_remove = [i*(spread + 2)     for i in range(nShots)] + \
                   [i*(spread + 2) + 1 for i in range(nShots)]

seismic = numpy.delete(seismic, traces_to_remove, axis = 1)

segyio.tools.from_array2D(f"{input_file}_raw.sgy", seismic.T)

data = mng.import_sgy_file(f"{input_file}_raw.sgy")

tsl = numpy.zeros(nShots * spread, dtype = int)
tsf = numpy.zeros(nShots * spread, dtype = int)

tsi = numpy.zeros(nShots * spread, dtype = int) + int(dt*1e6)
tsc = numpy.zeros(nShots * spread, dtype = int) + nt

src = numpy.zeros(nShots * spread, dtype = int)
rec = numpy.zeros(nShots * spread, dtype = int)

off = numpy.zeros(nShots * spread, dtype = int)
cmp = numpy.zeros(nShots * spread, dtype = int)
cmpx = numpy.zeros(nShots * spread, dtype = int)
cmpy = numpy.zeros(nShots * spread, dtype = int)

xsrc = numpy.zeros(nShots * spread, dtype = int)
ysrc = numpy.zeros(nShots * spread, dtype = int)
zsrc = numpy.zeros(nShots * spread, dtype = int)

xrec = numpy.zeros(nShots * spread, dtype = int)
yrec = numpy.zeros(nShots * spread, dtype = int)
zrec = numpy.zeros(nShots * spread, dtype = int)

gscal = numpy.zeros(nShots * spread, dtype = int) + scalar

spread_index = numpy.arange(spread)

for shot in range(nShots):

    fill = slice(shot*spread, shot*spread + spread)

    tsl[fill] = 1 + spread_index + shot*spread
    tsf[fill] = XPS[shot,0][:-10]
    
    src[fill] = SPS[shot,0][:-3]
    
    xsrc[fill] = float(SPS[shot,1])*scalar
    ysrc[fill] = float(SPS[shot,2])*scalar
    zsrc[fill] = float(SPS[shot,3])*scalar

    rec[fill] = numpy.array([k[:-3] for k in RPS[2*shot:2*shot+spread,0]], dtype = int)
    
    xrec[fill] = numpy.array([k for k in RPS[2*shot:2*shot+spread,1]], dtype = float)*scalar    
    yrec[fill] = numpy.array([k for k in RPS[2*shot:2*shot+spread,2]], dtype = float)*scalar    
    zrec[fill] = numpy.array([k for k in RPS[2*shot:2*shot+spread,3]], dtype = float)*scalar    

    off[fill] = numpy.array([int(x) for x in numpy.linspace(-0.5*spread, 0.5*spread, spread+1) if x != 0], dtype = int)*dr*scalar     
    
    cmp[fill] = numpy.arange(spread, dtype = int) + 2.0*(ds/dr)*shot + 1
    cmpx[fill] = xsrc[fill] - 0.5*(xsrc[fill] - xrec[fill]) 
    cmpy[fill] = ysrc[fill] - 0.5*(ysrc[fill] - yrec[fill]) 

bytes = [1, 5, 9, 13, 37, 21, 69, 115, 117, 
         41, 45, 73, 77, 81, 85, 181, 185]

values = [tsl, tsf, src, rec, off, cmp, gscal, 
          tsc, tsi, zrec, zsrc, xsrc, ysrc, 
          xrec, yrec, cmpx, cmpy]

mng.edit_trace_header(data, bytes, values)

mng.show_trace_header(data)
