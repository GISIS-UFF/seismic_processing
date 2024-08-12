import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filt
from toolbox import visualizing as view

input_name = "../data/GISIS_synthetic_data.sgy"

data = mng.import_sgy_file(input_name)

view.gather(data, key = "cmp", index = 800)
view.geometry(data, key = "cmp", index = 800)

view.fourier_fx_domain(data, key = "cmp", index = 800)
view.fourier_fk_domain(data, key = "cmp", index = 800)