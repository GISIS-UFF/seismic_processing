import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filt
from toolbox import visualizing as view

data = mng.import_sgy_file("../data/2D_Land_vibro_data_2ms/Line_001_raw.sgy")

data_muted = filt.mute(data)

view.gather(data_muted)

view.difference(data, data_muted)