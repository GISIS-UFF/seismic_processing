from toolbox import managing as mng

from toolbox import filtering as filter

from toolbox import visualizing as view

import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d



input_file = "data/overthrust_synthetic_seismic_data.sgy" # Synthetic data

input_file2 = "data/seismic.segy" #Mobil AVO viking graben line 12

input_file3 ="data/Line_001.sgy" #Poland 2D

input_file4 ="data/npr3_field.sgy" #Teapot dome 3D survey (HEAVY FILE)

input_file5 ="data/data_filt.sgy"

input_file6 ="data/difference"

input_file7 ="data/polandfilt"




data = mng.import_sgy_file(input_file3)


print(f'numero de traço {data.tracecount}')
mng.show_trace_header(data)

print(data.trace[0])
key = 'src'

indexes = mng.keyword_indexes(data, key)

print(indexes)

#index = 304
#view.gather(data, tlag = 0.1)
#datamute=mng.mute_traces(data,'testando',[0,1])
#view.gather(datamute, tlag = 0.1)
# view.fourier_fx_domain(data, key, index, fmin = 0, fmax = 100)


# fmin = 2    
# fmax = 50

# output_file = f"data/overthrust_seismic_data_{fmin}-{fmax}Hz.sgy"

# data_filt = filter.fourier_FX_domain(data, fmin, fmax, output_file)
 

# view.fourier_fx_domain(data_filt, key, index, fmin = 0, fmax = 100)

# view.difference(data, data_filt, key, index)

#mng.export_sgy_file(data_filt,'data_filt.sgy',)
#traces_to_remove = [0, 2]
#mng.extract_trace_gather(data,'polandfilt',traces_to_remove)

#seismic = data.trace.raw[:].T
    #seismic = seismic[:, traces]


def inter(data  , **kwargs) -> None: 




    

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
        fx_seismic[:,i] *= 1.0 / np.max(fx_seismic[:,i]) 

    scale = 0.8*np.std(seismic)

    mask = np.logical_and(frequency >= 0.0, frequency <= fmax)

    floc = np.linspace(0, len(frequency[mask])-1, 11, dtype = int)
    flab = np.around(np.ceil(frequency[floc]), decimals = 1)
    
    xloc = np.linspace(0, len(traces)-1, 5, dtype = int)
    xlab = traces[xloc]
    
    
    

    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (10, 5))
    points=[]
    def onclick(event):
        if event.inaxes is not None:
        # Captura o clique e adiciona o ponto à lista
            x, y = event.xdata, event.ydata
            points.append((x, y))
        # Desenha um ponto no gráfico
            
        else:
            print('Clique fora dos eixos.')

    
    def onkeypress(event):
        if event.key == 'enter':
            if len(points) > 1:
                points_sorted = sorted(points, key=lambda p: p[0])
                x, y = zip(*points_sorted)
            
            # Cria a interpolação linear
                interp_func = interp1d(x, y, kind='linear')
            
            # Gera novos pontos interpolados
                x_new = np.linspace(min(x), max(x), num=500)
                y_new = interp_func(x_new)
            
                plt.plot(x_new, y_new, 'b-')
                plt.draw()    

    

    fx = ax.imshow(np.abs(fx_seismic[mask,:]), aspect = "auto", cmap = "jet")
   
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


    plt.show()


inter(data)