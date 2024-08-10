import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

from toolbox import managing as mng

ponto_parada= None
interpolated_line = None

def interactive_velocitty_analysis(data : sgy.SegyFile, **kwargs):
    """
    Plot the velocity semblance of the according CMP Gather

    ### Mandatory Parameters:

    data: segyio object.

    ### Other Parameters

    index: CMP gather index. - First Complete CMP as Default
    
    vmin: minimum velocity in semblance. - 500.0 ms as Default
    
    vmax: maximum velocity in semblance. - 5000.0 ms as Default

    dv: velocity variation in semblance. - 250.0 ms as Default


    ### Examples:
    
    >>> view.semblance(data)
    >>> view.semblance(data, index=index, vmax=vmax)
    
    """

    key = 'cmp'

    byte = mng.__keywords.get(key) 

    _, cmps_per_traces = np.unique(data.attributes(byte)[:], return_counts = True)

    complete_cmp_indexes = np.where(cmps_per_traces == np.max(cmps_per_traces))[0]

    index = kwargs.get("index") if "index" in kwargs else complete_cmp_indexes[0] 

    mng.__check_index(data, key, index)

    vmin = kwargs.get("vmin") if "vmin" in kwargs else 500.0
    vmax = kwargs.get("vmax") if "vmax" in kwargs else 5000.0 
    dv = kwargs.get("dv") if "dv" in kwargs else 250.0

    traces = np.where(data.attributes(byte)[:] == index)[0]  
    
    seismic = data.trace.raw[:].T  
    seismic = seismic[:, traces]  

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    times = np.arange(nt) * dt
    velocities = np.arange(vmin, vmax+dv, dv)  
    
    nv = len(velocities)   

    offset = data.attributes(37)[traces] / data.attributes(69)[traces]

    semblance = np.zeros((nt, nv))
    points = []
    
    
    for i in range(nt): 
        for j in range(nv):  
            
            moveout = np.array(np.sqrt(times[i]**2 + (offset/velocities[j])**2) / dt, dtype = int)   

            mask = moveout < nt

            t = moveout[mask]
            x = np.arange(len(traces))[mask]
    
            semblance[i, j] += np.sum(np.abs(seismic[t, x]))**2         
    def onclick(event):

        
        global ponto_parada
        
        if event.inaxes is not None:
        
            x, y = event.xdata, event.ydata
            if ponto_parada is None:
                points.append((x, y))
                plt.plot(x, y, 'ro')
                plt.draw()
                print(points)
            else:
                points.append((x, y))
                plt.plot(x, y, 'bo')
                print(points)
            plt.draw()
            
            # for i in range (1,camadas):
            #     vint[i]=np.sqrt(((vrms[i])**2*points[i]-(vrms[i-1])**2*points[1])/(tt[i]-tt[i-1]))
        else:
            print('Clique fora dos eixos.')
    def on_key(event):
        if event.key=='n' and len(points) >= 1:
            index = int(input('Escolha o ponto para deletar (começando de 0): '))
            
            if 0 <= index < len(points):
                points.pop(index)
                ax[1].clear()
                im2 = ax[1].imshow(semblance, aspect='auto', cmap='jet', extent=[vmin, vmax, times[-1], times[0]])
                ax[1].set_xlabel('Velocity [m/s]', fontsize=15)
                ax[1].set_ylabel('Time [s]', fontsize=15)
                
                ax[1].set_xlim(vmin, vmax)

                ax[1].set_xlabel('Velocity [m/s]', fontsize=15)
                ax[1].set_ylabel('Time [s]', fontsize=15)
    
                
                
    
                for p in points:
                    plt.plot(p[0], p[1], 'ro')
                    plt.draw()
    def stop(event):
         global ponto_parada
         if event.key == 'j':
        
            if ponto_parada is None:  
                ponto_parada = (0, 0)
                
                plt.plot(ponto_parada[0], ponto_parada[1], 'go', label='Ponto de Parada')
                plt.draw()    
    
    def onkeypress(event):
    
        global interpolated_line

        if event.key == 'enter':
            if len(points) > 1:
                points_sorted = sorted(points, key = lambda p: p[1])
                x,y = zip(*points_sorted)
               
                


                cs = CubicSpline(y, x, bc_type='natural')
                ynew = np.linspace(min(y), max(y), num = 500)
                xnew = cs(ynew)
                if interpolated_line:
                    interpolated_line.remove()

                interpolated_line, = plt.plot(xnew, ynew)
                plt.draw()
        elif event.key =='m':
            if interpolated_line:
                interpolated_line.remove()  
                interpolated_line = None  
                plt.draw()
    def save(event):
        if event.key=='c':
            np.savetxt("coordernada", points, fmt = "%.6f")
            np.savetxt("coordernadainter", points, fmt = "%.6f")



    
    
    vmin_gather = np.percentile(seismic, 1)
    vmax_gather = np.percentile(seismic, 99) 

    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 5)) 

    im1 = ax[0].imshow(seismic, aspect='auto', cmap='Greys', vmin=vmin_gather, vmax=vmax_gather, extent=[0, seismic.shape[1], times[-1], times[0]])
    ax[0].set_xlabel('Trace number', fontsize=15)
    ax[0].set_ylabel('Time [s]', fontsize=15)
    cbar1 = fig.colorbar(im1, ax=ax[0])
    cbar1.set_label("Amplitude", fontsize=10)

    im2 = ax[1].imshow(semblance, aspect='auto', cmap='jet', extent=[vmin, vmax, times[-1], times[0]])
    ax[1].set_xlabel('Velocity [m/s]', fontsize=15)
    ax[1].set_ylabel('Time [s]', fontsize=15)
    cbar2 = fig.colorbar(im2, ax=ax[1])
    cbar2.set_label("Semblance", fontsize=10)

    ax[1].set_xlim(vmin, vmax)

    ax[1].set_xlabel('Velocity [m/s]', fontsize=15)
    ax[1].set_ylabel('Time [s]', fontsize=15)
    
    fig.suptitle(f'Velocity Semblance - Curent CMP: {index}', fontsize=16)

    fig.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkeypress)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('key_press_event', stop)
    fig.canvas.mpl_connect('key_press_event', save) 
    plt.show()
    

