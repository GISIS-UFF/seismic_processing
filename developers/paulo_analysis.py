import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import visualizing as view

data = mng.import_sgy_file("../data/2D_Land_vibro_data_2ms/Line_001_raw.sgy")

view.radon_transform(data, key = "rec", style = "hyperbolic")
