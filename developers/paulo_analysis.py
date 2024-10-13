import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import stacking as stk 
from toolbox import filtering as filt
from toolbox import visualizing as view

data = mng.import_sgy_file("../data/synthetic/synthetic_data_raw.sgy")

data_muted = filt.mute(data, time = 0.2, velocity = 1200)

view.gather(data_muted, key = "cmp")

cmps = mng.get_full_fold_cmps(data_muted)

cmps = cmps[::20]

cmps = 191

stk.interactive_velocity_analysis(data_muted, indexes = cmps)