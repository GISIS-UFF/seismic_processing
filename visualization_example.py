from toolbox import management as mng
from toolbox import visualization as view

data = mng.import_sgy_file("data/overthrust_synthetic_seismic_data.sgy")

mng.show_trace_header(data)

keyword = 'cmp'

indexes = view.keyword_indexes(data, keyword)

view.seismic(data, keyword, 478)
view.geometry(data, keyword, 478)

