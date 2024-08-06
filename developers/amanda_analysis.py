from sys import path
path.append("../")

from toolbox import visualizing as view
from toolbox import managing as mng

input_file = "../data/overthrust_synthetic_seismic_data.sgy" 

data = mng.import_sgy_file(input_file)

view.semblance(data)