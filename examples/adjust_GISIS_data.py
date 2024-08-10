import sys; sys.path.append("../")

from toolbox import managing as mng

import numpy

data1 = mng.import_sgy_file("../data/GISIS_synthetic_data_ex1.sgy")
data2 = mng.import_sgy_file("../data/GISIS_synthetic_data_ex2.sgy")

nShots = 201
spread = 469
scalar = 100

ds = 50.0

tsl = numpy.zeros(nShots * spread, dtype = int)
tsf = numpy.zeros(nShots * spread, dtype = int)

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
    tsf[fill] = shot + 1
    src[fill] = shot + 1
    
    xsrc[fill] = (6000 + shot*ds)*scalar
    ysrc[fill] = (6000 + shot*ds)*scalar
    zsrc[fill] = 0.0

    rec[fill] = [k+1 for k in range(spread)]
    
    off[fill] = numpy.array([x*scalar for x in numpy.linspace(150, 6000, spread)], dtype = int)     
    
    xrec[fill] = (off[fill][::-1] + shot*ds*scalar) - 15000   
    yrec[fill] = (off[fill][::-1] + shot*ds*scalar) - 15000   
    zrec[fill] = 0.0    

    cmp[fill] = numpy.arange(spread, dtype = int)[::-1] + 8*shot + 1
    
    cmpx[fill] = xsrc[fill] - 0.5*(xsrc[fill] - xrec[fill]) 
    cmpy[fill] = ysrc[fill] - 0.5*(ysrc[fill] - yrec[fill]) 

bytes = [1, 5, 9, 13, 21, 37, 41, 45, 69, 73, 77, 81, 85, 181, 185]

values = [tsl, tsf, src, rec, cmp, off, zrec, zsrc, gscal, 
          xsrc, ysrc, xrec, yrec, cmpx, cmpy]

mng.edit_trace_header(data1, bytes, values)
mng.edit_trace_header(data2, bytes, values)


