from sys import path
path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filt
from toolbox import visualizing as view

from time import time

input_name = "../data/overthrust_synthetic_seismic_data.sgy"

data = mng.import_sgy_file(input_name)

view.radon_transform(data, style = "hyperbolic")
