from sys import path
path.append("../")

#from toolbox import stacking as stk
from toolbox import managing as mng
from toolbox import visualizing as view

input_file = "../data/overthrust_synthetic_seismic_data.sgy" 


data = mng.import_sgy_file(input_file)
#cmps = [115, 150, 225, 360, 500]
#cmps = [225, 150]
#vmax = (6000)
pick_files = r"all_picks.txt"

# mng.get_full_fold_cmps(data)

#stk.interactive_velocity_analysis(data, indexes = cmps, vmax = vmax)

view.velocity_model(data, picks_file = pick_files) 
