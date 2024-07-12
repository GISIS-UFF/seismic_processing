from toolbox import managing as mng
from toolbox import filtering as filter
from toolbox import visualizing as view

input_file = "data/overthrust_synthetic_seismic_data.sgy" # Synthetic data
input_file2 = "data/seismic_data_Poland_line001_.sgy" #Mobil AVO viking graben line 12
input_file3 ="data/Line_001.sgy" #Poland 2D
input_file4 ="data/npr3_field.sgy" #Teapot dome 3D survey (HEAVY FILE)


data = mng.import_sgy_file(input_file2)

mng.show_trace_header(data)

key = 'src'

indexes = view.keyword_indexes(data, key)

print(indexes)

index = 286

view.gather(data, key, index)
# view.geometry(data, key, index)
# view.fourier_fx_domain(data, key, index, fmin = 0, fmax = 100)
# view.fourier_fk_domain(data, key, index, fmin = 0, fmax = 100)

# fmin = 2    
# fmax = 50

# output_file = f"data/overthrust_seismic_data_{fmin}-{fmax}Hz.sgy"

# data_filt = filter.fourier_FX_domain(data, fmin, fmax, output_file)

# view.fourier_fx_domain(data_filt, key, index, fmin = 0, fmax = 100)

# view.difference(data, data_filt, key, index)
