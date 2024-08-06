from sys import path
path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filt
from toolbox import visualizing as view

input_name = "../data/2D_Land_vibro_data_2ms/Line_001"

data = mng.import_sgy_file(input_name + "_raw.sgy")

view.fourier_fx_domain(data, trace_number = 100)

fmin = 5
fmax = 45

# filt.fourier_FX_domain(data, fmin = fmin, fmax = fmax, output_name = input_name + f"_bandpass_{fmin}-{fmax}Hz.sgy")

data_filt = mng.import_sgy_file(input_name + f"_bandpass_{fmin}-{fmax}Hz.sgy")

mng.show_trace_header(data_filt)

view.fourier_fx_domain(data_filt, trace_number = 100)

view.fourier_fk_domain(data_filt)
