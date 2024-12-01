import sys; sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

from toolbox import managing as mng
from toolbox import stacking as stk 
from toolbox import filtering as filt
from toolbox import visualizing as view

folder = "../data/synthetic/"
data_raw_file = "synthetic_data_raw.sgy"
data_mute_file =  "synthetic_data_mute.sgy"

data = mng.import_sgy_file(folder + data_raw_file)

cmps = mng.get_full_fold_cmps(data)

window = np.linspace(0, len(cmps)-1, 5, dtype = int)

cmpId = cmps[window]

stk.interactive_velocity_analysis(data, cmpId)

# seismic = filt.apply_agc(data_muted, time_window = 0.01, key = "cmp", index = cmpId[0])

# plt.imshow(seismic, aspect = "auto", cmap = "Greys")
# plt.show()
