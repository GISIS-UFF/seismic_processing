from sys import path
path.append("../")

from toolbox import managing as mng
from toolbox import visualizing as view

import numpy

input_file = "../data/mobil_avo_viking_graben_line_12/seismic.sgy"

data = mng.import_sgy_file(input_file)

print("Old trace header...\n")
mng.show_trace_header(data)

sx = data.attributes(73)[:]
sy = data.attributes(77)[:]

rx = data.attributes(81)[:]
ry = data.attributes(85)[:]

cmpx = numpy.array(sx + 0.5*(rx - sx), dtype = int)
cmpy = numpy.array(sy + 0.5*(ry - sy), dtype = int)

bytes = [181, 185]
values = [cmpx, cmpy]

mng.edit_trace_header(data, bytes, values)

print("\nNew trace header...\n")
mng.show_trace_header(data)

view.gather(data, key = "off", index = -262)
view.geometry(data, key = "cmp", index = 512) 
view.fourier_fx_domain(data, key = "cmp", index = 512) 
view.fourier_fk_domain(data, key = "cmp", index = 512) 
