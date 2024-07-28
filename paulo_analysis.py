from toolbox import managing as mng
from toolbox import visualizing as view

data = mng.import_sgy_file("data/2D_Land_vibro_data_2ms/Line_001_raw.sgy")

# view.gather(data)
# view.gather(data, key = "off")
# view.gather(data, key = "rec", index = 789)
view.gather(data, key = "cmp", index = 512)

# view.geometry(data)                           
# view.geometry(data, key = "off")              
# view.geometry(data, key = "rec", index = 789)   
view.geometry(data, key = "cmp", index = 512) 

# view.fourier_fx_domain(data, fmax = 200)                           
# view.fourier_fx_domain(data, key = "off")              
# view.fourier_fx_domain(data, key = "rec", index = 789)   
view.fourier_fx_domain(data, key = "cmp", index = 512) 

# view.fourier_fk_domain(data, fmax = 200)              
# view.fourier_fk_domain(data, key = "rec", index = 789)   
view.fourier_fk_domain(data, key = "cmp", index = 512) 
