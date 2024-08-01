import toolbox.visualizing as view
from toolbox import managing as mng

input_file = "data/overthrust_synthetic_seismic_data.sgy" # Synthetic data

data = mng.import_sgy_file(input_file)
key = 'cmp'
index = 286
style = 'linear'    

view.radon_transform(data, key, index, style)
