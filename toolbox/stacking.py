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

    all_picks = [] # list with all the picks

    for index in indexes:
        
        points=[]
        points_vxt=np.array([]).reshape(0, 3) # added column to put the cmp info

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
#        print(semblance.shape)

        def onclick(event):
            
            global stop_point
            
            if event.inaxes is not None:
            
                x, y = event.xdata, event.ydata
                if stop_point is None:
                    points.append((x, y))
                    nonlocal points_vxt
                    add= np.array([[index, velocities[int(x)], times[int(y)]]]) # added index
                    
                    points_vxt = np.append(points_vxt,add,axis=0)
                    print(points_vxt)
                    
                    
                    plt.plot(x, y, 'ro')
                    plt.draw()
                    print(velocities[int(x)], times[int(y)])    

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
                nonlocal points_vxt
                sorted_indices = np.argsort(points_vxt[:, 2])
                points_vxt = points_vxt[sorted_indices]
                
                all_picks.extend(points_vxt.tolist()) # add to all
                print(f"Picks for CMP {index} are saved.")
        
        def add(event):
            if event.key=='y':
                    nonlocal points_vxt
                    p0=points[0][0]
                    p1=points[-1][0]
                    #print(points_vxt)
                    first_point=points_vxt[0]
                    last_point=points_vxt[-1]

                    points.append((p0,min(tloc),))
                    points.append((p1,max(tloc)))
                    add_points = np.array([[index, last_point[1], max(times)], [index, first_point[1], min(times)]]) # added index
                    
                    points_vxt=np.concatenate((points_vxt,add_points),axis=0) 
                    
                    
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

    all_picks = np.array(all_picks)
    np.savetxt("all_picks.txt", all_picks, fmt="%d,%.4f,%.4f", delimiter=',')
    print("File 'all_picks.txt' saved with all picks.")


def apply_normal_moveout(data,index,picks):
    
    text=np.loadtxt(picks,delimiter=",")
    
    vrms= text[:,1].astype(float)
    
    tint=text[:,2].astype(float)
    
    byte = mng.__keywords.get('cmp')
    
    traces = np.where(data.attributes(byte)[:] == index)[0]
    
    nt = data.attributes(115)[0][0]
    
    dt = data.attributes(117)[0][0] * 1e-6
    
    
    x =data.attributes(37)[traces] /data.attributes(69)[traces]

    mng.__check_keyword('cmp')
    
    mng.__check_index(data, 'cmp', index)
    
    seismic = np.zeros((nt, len(traces)))

    
    print(len(traces),len(x))
    for i in range(len(traces)):
        seismic[:,i] = data.trace.raw[traces[i]] 
    
    model_nmo = np.zeros(nt) + vrms[-1]
    print(len(x))

    times = np.arange(nt) * dt
    for i in range(len(vrms)):
        model_nmo[int(np.sum(tint[:i]/dt)):int(np.sum(tint[:i+1]/dt))] = vrms[i]
        
        
        seismic_nmo = np.zeros_like(seismic)

    for i in range(nt):
	    
        curve = np.array(np.sqrt((i*dt)**2 + x**2 / model_nmo[i]**2) / dt, dtype = int) 	
            

        for j in range(len(x)):
            if curve[j]<nt:
                seismic_nmo[i,j]=seismic[curve[j],j]
        
       
		
    # tmute = x / 500
    # for j in range(len(x)):
    #     seismic_nmo[:int(tmute[j]/dt),j] = 0.0

    #trace = np.sum(seismic_nmo, axis = 1)
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 9))
    ax[0].plot(model_nmo, times)
    ax[0].set_ylim([0,5])
    ax[0].invert_yaxis()
    ax[1].imshow(seismic, aspect = "auto", cmap = "Greys")
    ax[1].set_xticks(np.linspace(0, len(traces), 5))
    ax[1].set_xticklabels(np.linspace(0, len(traces)-1, 5,dtype = int) )

    ax[1].set_yticks(np.linspace(0, nt, 11))
    ax[1].set_yticklabels(np.linspace(0, nt-1, 11)*dt)

    ax[1].set_title("CMP Gather", fontsize = 18)
    ax[1].set_xlabel("x = Offset [m]", fontsize = 15)
    ax[1].set_ylabel("t = TWT [s]", fontsize = 15)
    ax[2].imshow(seismic_nmo, aspect = "auto", cmap = "Greys")

    ax[2].set_xticks(np.linspace(0, len(traces), 5))
    ax[2].set_xticklabels(np.linspace(0, len(traces)-1, 5))

    ax[2].set_yticks(np.linspace(0, nt, 11))
    ax[2].set_yticklabels(np.linspace(0, nt-1, 11)*dt)

    ax[2].set_title("CMP Gather", fontsize = 18)
    ax[2].set_xlabel("x = Offset [m]", fontsize = 15)
    ax[2].set_ylabel("t = TWT [s]", fontsize = 15)

    # ax[3].plot(trace, time)
    # ax[3].set_ylim([0,5])
    # ax[3].invert_yaxis()
    plt.show()
        

def stack_cmp_gathers():
    # returns a post-stack seismic section 
    pass

