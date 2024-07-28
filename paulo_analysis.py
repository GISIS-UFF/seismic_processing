from toolbox import managing as mng
from toolbox import visualizing as view

import numpy

input_file = "data/mobil_avo_viking_graben_line_12.sgy"

data = mng.import_sgy_file(input_file)

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

# view.gather(data)
# view.gather(data, key = "off")
# view.gather(data, key = "rec", index = 789)
view.gather(data, key = "cmp", index = 512)

# view.geometry(data)                           
# view.geometry(data, key = "off")              
# view.geometry(data, key = "rec", index = 789)   
view.geometry(data, key = "cmp", index = 512) 

# view.fourier_fx_domain(data, fmax = 200)                           
# view.fourier_fx_domain(data, key = "off")              
# view.fourier_fx_domain(data, key = "rec", index = 789)   
# view.fourier_fx_domain(data, key = "cmp", index = 512) 

# view.fourier_fk_domain(data, fmax = 200)              
# view.fourier_fk_domain(data, key = "rec", index = 789)   
# view.fourier_fk_domain(data, key = "cmp", index = 512) 
