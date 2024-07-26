from toolbox import managing as mng
from toolbox import filtering as filter
from toolbox import visualizing as view

input_file = "data/seismic_data_Poland_line001.sgy"

data = mng.import_sgy_file(input_file)

mng.show_trace_header(data)

print(mng.keyword_indexes(data, key = "src"))

data_wind = mng.gather_windowing(data, "test_wind.sgy", 
                                 key = "src", index_beg = 231, index_end = 235)

mng.show_trace_header(data_wind)

data_mute = mng.mute_traces(data_wind, "test_mute.sgy", [1,2])

print(mng.keyword_indexes(data_mute, key = "src"))

view.difference(data_wind, data_mute, key = "src", index = 231)




# view.gather(data_mute, index = 232)
# view.geometry(data_mute)
# view.fourier_fx_domain(data_mute)
# view.fourier_fk_domain(data_mute)

# fmin = 2    
# fmax = 50

# output_file = f"data/overthrust_seismic_data_{fmin}-{fmax}Hz.sgy"

# data_filt = filter.fourier_FX_domain(data, fmin, fmax, output_file)

# view.fourier_fx_domain(data_filt, key, index, fmin = 0, fmax = 100)

# view.difference(data, data_filt, key, index)
