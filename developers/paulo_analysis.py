from sys import path
path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filt
from toolbox import visualizing as view

data = mng.import_sgy_file("../data/2D_Land_vibro_data_2ms/Line_001_raw.sgy")

view.fourier_fx_domain(data, trace_number = 200)
view.fourier_fx_domain(data, key = "off", trace_number = 200)
