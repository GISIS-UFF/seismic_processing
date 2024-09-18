import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filt
from toolbox import visualizing as view

data = mng.import_sgy_file("../data/2D_Land_vibro_data_2ms/Line_001_raw.sgy")

cmps = mng.get_full_fold_cmps(data)

data_cut = mng.gather_windowing(data, "full_fuld_data_test.sgy", key = "cmp", indexes_cut = cmps)

mng.show_trace_header(data_cut)

view.gather(data_cut, key = "cmp")