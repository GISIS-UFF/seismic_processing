import sys; sys.path.append("../")

from toolbox import managing as mng
from toolbox import modeling as mod
from toolbox import stacking as stk 
from toolbox import filtering as filt
from toolbox import visualizing as view

dh = 5.0
vmin = 2000.0
vmax = 2500.0

image_path = "../data/model.png"

model = mod.get_velocity_model_from_PNG(image_path, vmin, vmax)

dcmp = 25.0
cmp_min = 2000.0
cmp_max = 8000.0

model_traces = mod.get_cmp_traces_from_model(model, cmp_min, cmp_max, dcmp, dh)

dt = 2e-3
t_max = 4.0
f_max = 30.0
x_max = 4000.0  

cmp_gathers = mod.get_cmp_gathers(model_traces, t_max, f_max, x_max, dh, dt)

data_path = "../data/seismic.sgy"

data = mod.get_sgy_file(data_path, cmp_gathers, cmp_min, cmp_max, dcmp, dh, dt)

view.geometry(data, key = "cmp", index = 240)

