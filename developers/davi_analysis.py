from sys import path
import numpy as np
import matplotlib.pyplot as plt
from davi_functions import load_seismic, plot_seismic

path.append("../")
from toolbox import managing as mng

PATH = "../data/overthrust_synthetic_seismic_data" 

data = mng.import_sgy_file(PATH)

key = "rec"
index = 120 

seismic, tickNLabels = load_seismic(data, key, index)
seismic2 = np.copy(seismic)

dt = data.attributes(117)[0][0] * 1e-6
agc_operator = 10 # [ms]
sliding_window = int(agc_operator / (1e3 * dt))

nt, traces = seismic.shape

for i in range(traces):
    trace = seismic2[:, i]
    l, h = 0, sliding_window - 1
    mid = (l + h) // 2
    while h < nt - 1:
        window_samples = trace[l:h]
        mean_amplitude = np.mean(np.abs(window_samples))
        trace[mid] /= mean_amplitude + 1e-6

        l, h, mid = l + 1, h + 1, mid + 1

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 8))

scale = 0.9*np.std(seismic)
im = ax[0].imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)
ax[0].set_title("Normal Seismic")

scale2 = 0.9*np.std(seismic2)
im2 = ax[1].imshow(seismic2, aspect="auto", cmap="Greys", vmin=-scale2, vmax=scale2)
ax[1].set_title("AGC Seismic")

diff = seismic - seismic2
scale3 = 0.9*np.std(diff)
im3 = ax[2].imshow(diff, aspect="auto", cmap="Greys", vmin=-scale3, vmax=scale3)
ax[2].set_title("Diff A - B")

plt.show()

target_trace = 37
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

ax[0].plot(seismic[:, target_trace])
ax[0].set_title("Normal Trace")

ax[1].plot(seismic2[:, target_trace])
ax[1].set_title("AGC Trace")

for i in ax:
    i.set_ylim((min(seismic2[:, target_trace]), max(seismic2[:, target_trace])))
    
plt.show()
