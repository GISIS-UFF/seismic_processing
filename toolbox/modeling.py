import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

from toolbox import managing as mng

def __analytical_reflections(v, z, x):
    Tint = 2.0 * z / v[:-1]
    Vrms = np.zeros(len(z))
    reflections = np.zeros((len(z), len(x)))
    for i in range(len(z)):
        Vrms[i] = np.sqrt(np.sum(v[:i+1]**2 * Tint[:i+1]) / np.sum(Tint[:i+1]))
        reflections[i] = np.sqrt(x**2.0 + 4.0*np.sum(z[:i+1])**2) / Vrms[i]
    return reflections

def __wavelet_generation(nt, dt, fmax):
    ti = (nt/2)*dt
    fc = fmax / (3.0 * np.sqrt(np.pi)) 
    wavelet = np.zeros(nt)
    for n in range(nt):            
        arg = np.pi*((n*dt - ti)*fc*np.pi)**2    
        wavelet[n] = (1.0 - 2.0*arg)*np.exp(-arg);      
    return wavelet

def get_velocity_model_from_PNG(image_path, vmin, vmax):

    n = 256 

    image = plt.imread(image_path)[:,:,0]

    color = np.arange(n)/(n-1)
    model = np.zeros_like(image)
    velocity = color * (vmax - vmin) + vmin

    for k in range(n):
        model[np.where(color[k] == image)] = velocity[n-k-1]

    return model

def get_cmp_traces_from_model(model, cmp_min, cmp_max, dcmp, dh):

    nCMP = int((cmp_max - cmp_min) / dcmp) + 1    

    x = np.linspace(cmp_min/dh, cmp_max/dh, nCMP, dtype = int)

    return model[:,x] 

def get_cmp_gathers(model_traces, t_max, f_max, x_max, dh, dt):

    nz, nCMP = np.shape(model_traces)

    nt = int(t_max / dt) + 1
    nr = int(x_max / dh) + 1

    depth = np.arange(nz)*dh
    offset = np.arange(nr)*dh

    wavelet = __wavelet_generation(nt, dt, f_max)

    cmp_gathers = np.zeros((nt, nr*nCMP))

    for cmpId in range(nCMP):

        print(f"Running CMP {cmpId+1} of {nCMP}")    

        velocity = model_traces[:, cmpId]    

        reflectivity = (velocity[1:] - velocity[:-1]) / (velocity[1:] + velocity[:-1])    
        
        i = np.where(reflectivity > 0)[0][::2]

        z = depth[i]
        v = velocity[i]
        v = np.append(v, velocity[-1])

        reflections = __analytical_reflections(v, z, offset)

        seismogram = np.zeros((nt, nr))

        for j in range(nr):
            for i in range(len(z)):    
                indt = int(reflections[i, j] / dt)
                if indt < nt:    
                    seismogram[indt, j] = 1.0

            seismogram[:,j] = np.convolve(seismogram[:, j], wavelet, "same")

        cmp_gathers[:, cmpId*nr:cmpId*nr + nr] = seismogram

    return cmp_gathers

def get_sgy_file(data_path, cmp_gathers, cmp_min, cmp_max, dcmp, dh, dt):

    nt = len(cmp_gathers)
    nTraces = len(cmp_gathers[0])
    nCMP = int((cmp_max - cmp_min) / dcmp) + 1    
    cmps = np.linspace(cmp_min, cmp_max, nCMP)
   
    nr = int(nTraces / nCMP)  
    cmpi = np.arange(nr) + 1
    offset = np.arange(nr) * dh

    tsl = np.zeros(nTraces, dtype = int)
    tsf = np.zeros(nTraces, dtype = int)

    tsi = np.zeros(nTraces, dtype = int) + int(dt*1e6)
    tsc = np.zeros(nTraces, dtype = int) + nt

    off = np.zeros(nTraces, dtype = int)
    cmp = np.zeros(nTraces, dtype = int)
    cmpx = np.zeros(nTraces, dtype = int)

    xsrc = np.zeros(nTraces, dtype = int)
    xrec = np.zeros(nTraces, dtype = int)

    gscal = np.zeros(nTraces, dtype = int) + 100

    for cmpId in range(nCMP):
        
        fill = slice(cmpId*nr, cmpId*nr + nr)

        tsl[fill] = cmpi*(cmpId + 1)
        tsf[fill] = cmpi
    
        off[fill] = offset     
        cmp[fill] = cmpi[cmpId]
        cmpx[fill] = cmps[cmpId]*gscal[fill] 
    
        xsrc[fill] = (cmps[cmpId] + 0.5*offset)*gscal[fill]
        xrec[fill] = (cmps[cmpId] - 0.5*offset)*gscal[fill]

    sgy.tools.from_array2D(data_path, cmp_gathers.T)
    data = mng.import_sgy_file(data_path)

    bytes = [1, 5, 37, 21, 69, 115, 117, 73, 81, 181]

    values = [tsl, tsf, off, cmp, gscal, tsc, tsi, xsrc, xrec, cmpx]

    mng.edit_trace_header(data, bytes, values)

    mng.show_trace_header(data)

    return data

