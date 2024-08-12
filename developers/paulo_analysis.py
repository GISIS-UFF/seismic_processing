import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filt
from toolbox import visualizing as view

# input_name = "../data/2D_Land_vibro_data_2ms/Line_001_raw.sgy"
input_name = "../data/GISIS_synthetic_data.sgy"

data = mng.import_sgy_file(input_name)

mng.show_trace_header(data)

view.gather(data, key = "rec", index = 680)
view.geometry(data, key = "rec", index = 680)

# view.fourier_fx_domain(data, key = "cmp", index = 800, trace_number = 50)
# view.fourier_fk_domain(data, key = "cmp", index = 800)

# view.radon_transform(data, index = 800, style = "linear")
# view.radon_transform(data, index = 800, style = "parabolic")
# view.radon_transform(data, index = 800, style = "hyperbolic")