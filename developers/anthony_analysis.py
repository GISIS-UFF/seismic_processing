from sys import path
path.append("../")

from toolbox import managing as mng
from toolbox import filtering as filter
from toolbox import visualizing as view

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

input_file = "../data/overthrust_synthetic_seismic_data.sgy"  # Synthetic data
input_file2 = "../data/seismic.segy"  # Mobil AVO viking graben line 12
input_file3 = "../data/Line_001.sgy"  # Poland 2D
input_file4 = "../data/npr3_field.sgy"  # Teapot dome 3D survey (HEAVY FILE)
input_file5 = "../data/data_filt.sgy"
input_file6 = "../data/difference"
input_file7 = "../data/polandfilt"

data = mng.import_sgy_file(input_file)

print(f'numero de traço {data.tracecount}')
mng.show_trace_header(data)

print(data.trace[0])
key = 'src'

indexes = mng.keyword_indexes(data, key)

print(indexes)

index = 304
interpolated_line = None
ponto_parada= None
def inter(data, **kwargs) -> None:
    fmax = kwargs.get("fmax") if "fmax" in kwargs else 100.0
    key = kwargs.get("key") if "key" in kwargs else "src"
    index = kwargs.get("index") if "index" in kwargs else mng.keyword_indexes(data, key)[0] 

    mng.__check_keyword(key)
    mng.__check_index(data, key, index)

    byte = mng.__keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]    

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    frequency = np.fft.fftfreq(nt, dt)
    fx_seismic = np.fft.fft(seismic, axis = 0)

    for i in range(len(traces)):
        fx_seismic[:, i] *= 1.0 / np.max(fx_seismic[:, i]) 

    scale = 0.8 * np.std(seismic)

    mask = np.logical_and(frequency >= 0.0, frequency <= fmax)

    floc = np.linspace(0, len(frequency[mask]) - 1, 11, dtype = int)
    flab = np.around(np.ceil(frequency[floc]), decimals = 1)

    xloc = np.linspace(0, len(traces) - 1, 5, dtype = int)
    xlab = traces[xloc]

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (10, 5))
    points = []
    


    def onclick(event):
        global ponto_parada
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            if ponto_parada is None:
                points.append((x, y))
                plt.plot(x, y, 'ro')
                plt.draw()
            else:
                points.append((x, y))
                plt.plot(x, y, 'bo')
            plt.draw()
        else:
            print('Clique fora dos eixos.')
    def stop(event):
         global ponto_parada
         if event.key == 'j':
        
            if ponto_parada is None:  
                ponto_parada = (0, 0)
                
                plt.plot(ponto_parada[0], ponto_parada[1], 'go', label='Ponto de Parada')
                plt.draw()

    def on_key(event):
        if event.key=='n' and len(points) >= 1:
            index = int(input('Escolha o ponto para deletar (começando de 0): '))
            if 0 <= index < len(points):
                points.pop(index)
                ax.clear()
                ax.imshow(np.abs(fx_seismic[mask, :]), aspect="auto", cmap="jet")
                ax.set_yticks(floc)
                ax.set_yticklabels(flab)
                ax.set_xticks(xloc)
                ax.set_xticklabels(xlab)
                for p in points:
                    plt.plot(p[0], p[1], 'ro')
                    plt.draw()
                

   

    def onkeypress(event):
        global interpolated_line
        if event.key == 'enter':
            if len(points) > 1:
                points_sorted = sorted(points, key = lambda p: p[0])
                x, y = zip(*points_sorted)
                

                interp_func = interp1d(x, y, kind = 'linear')

                x_new = np.linspace(min(x), max(x), num = 500)
                y_new = interp_func(x_new)
                if interpolated_line:
                    interpolated_line.remove()

                interpolated_line, = plt.plot(x_new, y_new, 'b-')
                plt.draw()
        elif event.key =='m':
            if interpolated_line:
                interpolated_line.remove()  
                interpolated_line = None  
                plt.draw()


    fx = ax.imshow(np.abs(fx_seismic[mask, :]), aspect = "auto", cmap = "jet")

    ax.set_yticks(floc)
    ax.set_yticklabels(flab)
    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)

    ax.set_ylabel("Frequency [Hz]", fontsize = 15)
    ax.set_xlabel("Trace number", fontsize = 15)

    ax.cbar = fig.colorbar(fx, ax = ax)
    ax.cbar.set_label("Amplitude", fontsize = 15) 

    fig.tight_layout()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkeypress)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('key_press_event', stop)

    plt.show()

#inter(data)

view.semblance(data)
# P=np.loadtxt('coordernada',dtype = float)
# print(len(P))
# vint=np.zeros(len(P))
# vint[0]=P[0,0]

# Prof=np.zeros(len(P))
# Prof[0] = 0.5*P[1,0]*vint[0]
# for i in range (1,len(P)):
#     vint[i]=np.sqrt(((P[i,0])**2*P[i,1]-(P[i-1,0])**2*P[i-1,1])/(P[i,1]-P[i-1,1]))
#     Prof[i] = Prof[i-1] + 0.5*(P[i,1]-P[i-1,1])*vint[i]
# print(vint)
# print(Prof)