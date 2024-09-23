import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filt
from toolbox import visualizing as view

data = mng.import_sgy_file("../data/mobil_viking_graben_north_sea.sgy")

# mng.show_trace_header(data)

cmps = mng.get_full_fold_cmps(data)

cmps = cmps[0]

data_cut = mng.gather_windowing(data, "full_fuld_data_test.sgy", key = "cmp", indexes_cut = cmps)

mng.show_trace_header(data_cut)

print(mng.get_keyword_indexes(data_cut, key = "cmp"))

view.gather(data, key = "cmp")