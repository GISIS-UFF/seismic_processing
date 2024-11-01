import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
# from toolbox import filtering

def butter_bandpass_filter(input_trace, lowcut, highcut, fs, order=6):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return lfilter(b, a, input_trace)

def time_variant_filtering(signal, lowcut_freqs, highcut_freqs, durations, fs):
    """
    Applies time-varying filtering to a signal.

    Parameters:
        signal (numpy array): Entry signal.
        lowcut_freqs (list): Lower cutoff frequencies for each range.
        highcut_freqs (list): Upper cutoff frequencies for each range.
        durations (list): Duration of intervals
        fs (int): Sampling rate.

    Returns:
        filtered_signal (numpy array): Filtered signal.
    """
    nt = len(signal)  
    
    filtered_signal = np.zeros(nt) 

    # Calculate the start and end times of intervals
    start_times = [0] + np.cumsum(durations).tolist()[:-1]  # Start of intervals
    end_times = np.cumsum(durations).tolist()  # End of intervals

    # Filtering the signal at each interval using a loop
    for i in range(len(lowcut_freqs)):
        start_sample = int(start_times[i] * fs)  # Time Conversion for Sample Index
        end_sample = int(end_times[i] * fs) if i < len(end_times) else nt  # Checks if the index is within the limit
        # Apply the filter to the range
        filtered_signal[start_sample:end_sample] = butter_bandpass_filter( signal[start_sample:end_sample], lowcut_freqs[i], highcut_freqs[i], fs)

    return filtered_signal

# Example of using the function
if __name__ == "__main__":
    signal = np.loadtxt('../data/trace_test.txt')

    fs = 512  
    nt = len(signal)
    t = np.arange(nt) * 1/fs

    # Cutoff frequencies for different ranges
    lowcut_freqs = [10, 20, 30]  # Minimum frequencies for each interval
    highcut_freqs = [200, 100, 50]  # Maximum frequencies for each interval

    # Duration of intervals (in seconds)
    durations = [3, 2, 5]  # 0-3s, 3-5s, 5-10s

    # time_variant_filtering
    filtered_signal = time_variant_filtering(signal, lowcut_freqs, highcut_freqs, durations, fs)

    # FFT
    freq = np.fft.fftfreq(nt, 1/fs)
    mask = freq > 0
    
    fft_signal = np.abs(np.fft.fft(signal))
    fft_filtered_signal = np.abs(np.fft.fft(filtered_signal))


    # Plot
    plt.figure(figsize=(12, 7))

    plt.subplot(221)
    plt.plot(t, signal, label=' Original Trace')
    plt.title('Original Trace')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(freq[mask], fft_signal[mask], label=' Original Trace FFT')
    plt.title(' Original Trace FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.xlim(0,150)

    plt.subplot(223)
    plt.plot(t, filtered_signal, label='Filtered Trace')
    plt.title('Filtered Trace')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(224)
    plt.plot(freq[mask], fft_filtered_signal[mask], label='Filtered Trace FFT')
    plt.title('FFT do Sinal Filtrado')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0,150)

    plt.tight_layout()
    plt.show()



# from sys import path
# path.append("../")

# from toolbox import managing as mng
# from toolbox import filtering
# from toolbox import visualizing as view

# data = mng.import_sgy_file("../data/Buzios_2D_streamer_6Km_GISIS_dada_broadbandwave.segy")

# velocity = 1800
# t0 = 0.8

# filtering.mute(data, velocity, t0)

# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------


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

# seismogram = np.zeros((nt, nx), dtype = np.float32)
# wavelet = wavelet_generation(nt, dt, fmax)

# for j in range(nx):
#     for i in range(len(z)):
#         indt = int(reflections[i, j] / dt)
#         seismogram[indt, j] = 1.0

#     seismogram[:,j] = np.convolve(seismogram[:, j], wavelet, "same")

# sgy.tools.from_array2D("../data/seismic_radon_test.sgy", seismogram.T)

# data = mng.import_sgy_file("../data/seismic_radon_test.sgy")

# scalar = 100

# tsl = np.arange(nx) + 1
# tsf = np.ones(nx)

# tsi = np.zeros(nx) + int(dt*1e6)
# tsc = np.zeros(nx) + nt

# src = np.zeros(nx) + 1001
# rec = np.arange(nx) + 2001

# off = np.arange(nx) * dx * scalar

# cmp = np.zeros(nx) + 1

# xsrc = np.zeros(nx)
# ysrc = np.zeros(nx)
# zsrc = np.zeros(nx)

# xrec = np.arange(nx) * dx * scalar
# yrec = np.arange(nx) * dx * scalar
# zrec = np.zeros(nx)

# cmpx = xsrc - 0.5*(xsrc - xrec)*scalar
# cmpy = ysrc - 0.5*(ysrc - yrec)*scalar

# gscal = np.zeros(n_receivers, dtype = int) + scalar

# bytes = [1, 5, 9, 13, 37, 21, 69, 115, 117,
#          41, 45, 73, 77, 81, 85, 181, 185]

# values = [tsl, tsf, src, rec, off, cmp, gscal,
#           tsc, tsi, zrec, zsrc, xsrc, ysrc,
#           xrec, yrec, cmpx, cmpy]

# mng.edit_trace_header(data, bytes, values)

# mng.show_trace_header(data)

# view.gather(data)
# view.geometry(data)
# view.radon_transform(data, style = "hyperbolic", index = 1, vmin = 1000, vmax = 3000)

# # data2 = mng.import_sgy_file("../data/overthrust_synthetic_seismic_data.sgy")
# data2 = mng.import_sgy_file("../data/seismic_radon_test.sgy")


# style = 'hyperbolic'
# key = 'cmp'
# index = 1
# filtering.radon_transform2(data2, key, index, style)

