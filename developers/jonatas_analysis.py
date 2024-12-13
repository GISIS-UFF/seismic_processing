from sys import path
path.append("../")

import numpy as np
import matplotlib.pyplot as plt

from toolbox import managing as mng
from toolbox import filtering
    
# Example

input_file1 = "../data/overthrust_synthetic_seismic_data.sgy" 
# input_file1 = "../data/Buzios_2D_streamer_6Km_GISIS_data_realwave.segy" 

seismic = mng.import_sgy_file(input_file1)
filtering.time_variant_filtering(seismic, num_windows=5)
