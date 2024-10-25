import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import stacking as stk 
from toolbox import filtering as filt
from toolbox import visualizing as view

data = mng.import_sgy_file("../data/synthetic/synthetic_data_raw.sgy")

data_muted = filt.mute(data, time = 0.2, velocity = 800)

view.gather(data_muted, key = "cmp")

cmps = 191

stk.interactive_velocity_analysis(data_muted, indexes = cmps)