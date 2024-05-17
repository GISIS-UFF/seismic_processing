from toolbox import managing as mng
from toolbox import filtering as filter
from toolbox import visualizing as view

data = mng.import_sgy_file("data/overthrust_synthetic_seismic_data.sgy")

mng.show_trace_header(data)

key = 'src'

indexes = view.keyword_indexes(data, key)

index = 1

view.seismic(data, key, index)
view.geometry(data, key, index)
view.fourier_fx_domain(data, key, index, fmin = 0, fmax = 100)
view.fourier_fk_domain(data, key, index, fmin = 0, fmax = 100)

