import numpy as np

def get_velocity_model_from_PNG(image):
    return None

def get_cmp_traces_from_model(model):
    
    # x position array of each trace
      
    return None

def get_cmp_gathers(traces):

    # x_max and t_max definition

    # Commom mid point modeling example

    # def analytical_reflections(v, z, x):
    #     Tint = 2.0 * z / v[:-1]
    #     Vrms = np.zeros(len(z))
    #     reflections = np.zeros((len(z), len(x)))
    #     for i in range(len(z)):
    #         Vrms[i] = np.sqrt(np.sum(v[:i+1]**2 * Tint[:i+1]) / np.sum(Tint[:i+1]))
    #         reflections[i] = np.sqrt(x**2.0 + 4.0*np.sum(z[:i+1])**2) / Vrms[i]
    #     return reflections

    # def wavelet_generation(nt, dt, fmax):
    #     ti = (nt/2)*dt
    #     fc = fmax / (3.0 * np.sqrt(np.pi)) 
    #     wavelet = np.zeros(nt)
    #     for n in range(nt):            
    #         arg = np.pi*((n*dt - ti)*fc*np.pi)**2    
    #         wavelet[n] = (1.0 - 2.0*arg)*np.exp(-arg);      
    #     return wavelet

    # n_receivers = 320
    # spread_length = 8000
    # total_time = 5.0
    # fmax = 30.0

    # dx = 25
    # dt = 1e-3

    # nt = int(total_time / dt) + 1
    # nx = int(n_receivers / 2) + 1

    # z = np.array([500, 1000, 1000, 1000])
    # v = np.array([1500, 1650, 2000, 3000, 4500])

    # x = np.linspace(0, nx*dx, nx)

    # reflections = analytical_reflections(v, z, x)

    # seismogram = np.zeros((nt, nx))
    # wavelet = wavelet_generation(nt, dt, fmax)

    # for j in range(nx):
    #     for i in range(len(z)):
    #         indt = int(reflections[i, j] / dt)
    #         seismogram[indt, j] = 1.0

    #     seismogram[:,j] = np.convolve(seismogram[:, j], wavelet, "same")
    return None

def export_sgy_file(cmps):
    return None

