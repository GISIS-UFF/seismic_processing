import sys; sys.path.append("../")

from toolbox import managing as mng

import numpy

input_file = "../data/mobil_viking_graben_north_sea.sgy"

data = mng.import_sgy_file(input_file)

sx = data.attributes(73)[:]
sy = data.attributes(77)[:]

rx = data.attributes(81)[:]
ry = data.attributes(85)[:]

cmpx = numpy.array(sx + 0.5*(rx - sx), dtype = int)
cmpy = numpy.array(sy + 0.5*(ry - sy), dtype = int)

bytes = [181, 185]
values = [cmpx, cmpy]

mng.edit_trace_header(data, bytes, values)
