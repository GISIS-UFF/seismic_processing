import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

from toolbox import managing as mng
from toolbox import filtering as filt

stop_point= None
interpolated_line = None


def interactive_velocity_analysis(data : sgy.SegyFile, indexes : np.ndarray, **kwargs):
    """
    Perform an interactive velocity analysis.

    ### Mandatory Parameters:

    data: segyio object.

    indexes: CMPs for analysis.

    ### Other Parameters
    
    vmin: minimum velocity in semblance.
    
    vmax: maximum velocity in semblance.

    nvel: total velocity parameters.

    ### Examples:
    
    >>> stk.interactive_velocity_analysis(data, indexes = cmps)

    ### Tutorial for picking velocity:
    
    1) Click on the   semblance graphic.
    2) After placing the points, press 'y' and then 'Enter'.
    3) Save is currently under maintenance.
    
    This function explains the steps needed for velocity picking.

    ### Returns:
    
    parameters.txt  (t,v)
    
    
    """

    key = 'cmp'

    byte = mng.__keywords.get(key) 

    indexes = [indexes] if not isinstance(indexes, (np.ndarray, list)) else indexes 

    mng.__check_index(data, key, indexes)
    
    vmin = kwargs.get("vmin") if "vmin" in kwargs else 1000.0
    vmax = kwargs.get("vmax") if "vmax" in kwargs else 3000.0 
    nvel = kwargs.get("dv") if "dv" in kwargs else 101

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    times = np.arange(nt) * dt
    velocities = np.linspace(vmin, vmax, nvel)
    for k in indexes:
        points=[]
        print(k)
        for index in [k]:

            traces = np.where(data.attributes(byte)[:] == index)[0]
            
            
            offsets = data.attributes(37)[traces] / data.attributes(69)[traces]
            
            seismic = np.zeros((nt, len(traces)))

            gain = times ** 2

            for i in range(len(traces)):
                seismic[:,i] = data.trace.raw[traces[i]] * gain

            domain = filt.radon_forward(seismic, dt, times, offsets, velocities, style = "hyperbolic")
        
            semblance = np.sum(np.abs(domain)**2, axis = 1)

            semblance = gaussian_filter(semblance, sigma = 3.0)

            semblance = np.abs(np.gradient(semblance, axis = 0))

            semblance = gaussian_filter(semblance, sigma = 5.0)
            print(semblance.shape)

        def onclick(event):
            
            global stop_point
            
            if event.inaxes is not None:
            
                x, y = event.xdata, event.ydata
                if stop_point is None:
                    points.append((x, y))
                    print(points)
                    plt.plot(x, y, 'ro')
                    plt.draw()
                    
                else:
                    points.append((x, y))
                    plt.plot(x, y, 'bo')
                    
                plt.draw()
                
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
        
        
        #def stop(event):
            #  global ponto_parada
            #  if event.key == 'j':
            
            #     if ponto_parada is None:  
            #         ponto_parada = (0, 0)
                    
            #         plt.plot(ponto_parada[0], ponto_parada[1], 'go', label='Ponto de Parada')
            #         plt.draw()    
        

        def onkeypress(event):
        
            global interpolated_line

            if event.key == 'enter':
                if len(points) > 1:
                    points_sorted = sorted(points, key = lambda p: p[1])
                    x,y = zip(*points_sorted)
                
                    cs = interp1d(y, x, kind='linear')
                    ynew = np.linspace(min(y), max(y), num = 500)
                    xnew = cs(ynew)
                    xsmooth = gaussian_filter(xnew, sigma=7)
                    if interpolated_line:
                        interpolated_line.remove()

                    interpolated_line, = plt.plot(xsmooth, ynew)
                    plt.draw()

            elif event.key =='m':
                if interpolated_line:
                    interpolated_line.remove()  
                    interpolated_line = None  
                    plt.draw()
        
        
        def save(event):
            if event.key=='c':
                np.savetxt("parameters.txt", points, fmt = "%.6f")
        
        
        def add(event):
            if event.key=='y':
                    p0=points[0][0]
                    p1=points[-1][0]
                    

                    points.append((p0,min(tloc),))
                    points.append((p1,max(tloc)))
                    print(points)
                    xn = [p[0] for p in points]
                    yn = [p[1] for p in points]
                    plt.plot(xn,yn,'ro')
                    plt.draw()
                    
        xloc = np.linspace(0, len(traces)-1, 5, dtype = int)

        tloc = np.linspace(0, nt-1, 11, dtype = int)
        tlab = np.around(tloc * dt, decimals = 3)
        
        vloc = np.linspace(0, nvel-1, 5, dtype = int)
        vlab = velocities[vloc]

        fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 5)) 

        im1 = ax[0].imshow(seismic, aspect = 'auto', cmap = 'Greys')
        ax[0].set_xticks(xloc)
        ax[0].set_yticks(tloc)
        ax[0].set_xticklabels(xloc)
        ax[0].set_yticklabels(tlab)
        ax[0].set_xlabel('Relative trace index', fontsize = 15) 
        ax[0].set_ylabel('Time [s]', fontsize = 15)
        cbar1 = fig.colorbar(im1, ax = ax[0])
        cbar1.set_label("Amplitude", fontsize = 15)

        im2 = ax[1].imshow(semblance, aspect = 'auto', cmap = 'jet')
        ax[1].set_xlabel('Velocity [m/s]', fontsize = 15) 
        ax[1].set_ylabel('Time [s]', fontsize = 15)
        ax[1].grid()
        ax[1].set_xticks(vloc)
        ax[1].set_yticks(tloc)
        ax[1].set_xticklabels(vlab)
        ax[1].set_yticklabels(tlab)
        ax[1].set_xlabel('Velocity [m/s]', fontsize = 15) 
        ax[1].set_ylabel('Time [s]', fontsize = 15)
        cbar2 = fig.colorbar(im2, ax = ax[1])
        cbar2.set_label("Amplitude", fontsize = 10)

        
        
        fig.tight_layout()
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkeypress)
        fig.canvas.mpl_connect('key_press_event', on_key)
        #fig.canvas.mpl_connect('key_press_event', stop)
        fig.canvas.mpl_connect('key_press_event', save)
        fig.canvas.mpl_connect('key_press_event', add)  
        plt.show()
    
def apply_normal_moveout():
    # returns flat pre-stack cmp gathers
    pass

def stack_cmp_gathers():
    # returns a post-stack seismic section 
    pass

