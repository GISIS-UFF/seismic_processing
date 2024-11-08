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
    # signal = np.loadtxt('../data/trace_test.txt')

    # fs = 512  
    # nt = len(signal)
    # t = np.arange(nt) * 1/fs

    nt = 6001
    nx = 321
    dt = 0.001
    fs = 1/dt

    data = np.reshape(np.fromfile('../data/elastic_iso_data_nStations321_nSamples6001_shot_6.bin', dtype=np.float32), (nx, nt))

    # Aplicando o filtro no seismogram
    data_filt = np.zeros_like(data)

    for i in range(len(data[:,0])):
            data_filt[i,:] = butter_bandpass_filter(data[i,:], lowcut= 5, highcut=300, fs=fs)


    # Cutoff frequencies for different ranges
    lowcut_freqs = [20, 10, 5]  # Minimum frequencies for each interval
    highcut_freqs = [250, 230, 210]  # Maximum frequencies for each interval

    # Duration of intervals (in seconds)
    durations = [2, 2, 4]  # 0-3s, 3-5s, 5-10s

    data_filt2 = np.zeros_like(data)

    for i in range(len(data[:,0])):
            data_filt2[i,:] = time_variant_filtering(data[i,:], lowcut_freqs, highcut_freqs, durations, fs)

    # time_variant_filtering
    
    scale = np.percentile(data, 95)
    scale_filt = np.percentile(data_filt, 95)
    scale_filt2 = np.percentile(data_filt2, 95)

    plt.figure()

    plt.subplot(131)
    plt.imshow(data.T, aspect='auto', cmap='Grays', vmin=-scale, vmax=scale)

    plt.subplot(132)
    plt.imshow(data_filt.T, aspect='auto', cmap='Grays', vmin=-scale_filt, vmax=scale_filt)

    plt.subplot(133)
    plt.imshow(data_filt2.T, aspect='auto', cmap='Grays', vmin=-scale_filt2, vmax=scale_filt2)

    plt.tight_layout()

    plt.show()


    exit()
    trace = data[25,:]
    time = np.arange(nt) * dt
    

    # Cutoff frequencies for different ranges
    lowcut_freqs = [10, 20, 190]  # Minimum frequencies for each interval
    highcut_freqs = [200, 100, 200]  # Maximum frequencies for each interval

    # Duration of intervals (in seconds)
    durations = [1, 1, 4]  # 0-3s, 3-5s, 5-10s

    # time_variant_filtering
    filtered_trace = time_variant_filtering(trace, lowcut_freqs, highcut_freqs, durations, fs)

    # FFT
    freq = np.fft.fftfreq(nt, dt)
    mask = freq > 0
    
    fft_trace = np.abs(np.fft.fft(trace))
    fft_filtered_trace = np.abs(np.fft.fft(filtered_trace))


    # Plot
    plt.figure(figsize=(12, 7))

    plt.subplot(221)
    plt.plot(time, trace, label=' Original Trace')
    plt.title('Original Trace')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(222)
    plt.plot(freq[mask], fft_trace[mask], label=' Original Trace FFT')
    plt.title(' Original Trace FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.xlim(0,150)

    plt.subplot(223)
    plt.plot(time, filtered_trace, label='Filtered Trace')
    plt.title('Filtered Trace')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(224)
    plt.plot(freq[mask], fft_filtered_trace[mask], label='Filtered Trace FFT')
    plt.title('FFT do Sinal Filtrado')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0,150)

    plt.tight_layout()
    plt.show()


    ### TESTANDO O CÓDIGO

    trace1 = trace[0:2000]
    trace2 = trace[2000:4000]
    trace3 = trace[4000:]

    filtered_trace1 = filtered_trace[0:2000]
    filtered_trace2 = filtered_trace[2000:4000]
    filtered_trace3 = filtered_trace[4000:]

    freq1 = np.fft.fftfreq(len(trace1), dt)
    mask1 = freq1 > 0
    
    fft_trace1 = np.abs(np.fft.fft(trace1))
    fft_filtered_trace1 = np.abs(np.fft.fft(filtered_trace1))


    freq2 = np.fft.fftfreq(len(trace2), dt)
    mask2 = freq2 > 0
    
    fft_trace2 = np.abs(np.fft.fft(trace2))
    fft_filtered_trace2 = np.abs(np.fft.fft(filtered_trace2))

    freq3 = np.fft.fftfreq(len(trace3), dt)
    mask3 = freq3 > 0
    
    fft_trace3 = np.abs(np.fft.fft(trace3))
    fft_filtered_trace3 = np.abs(np.fft.fft(filtered_trace3))


    plt.figure()
    
    plt.subplot(231)
    plt.title('1º Intervalo Bruto')
    plt.plot(time[0:2000], trace1)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(232)
    plt.title('2º Intervalo Bruto')
    plt.plot(time[2000:4000], trace2)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(233)
    plt.title('3º Intervalo bruto')
    plt.plot(time[4000:], trace3)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(234)
    plt.title('1º Intervalo filt')
    plt.plot(time[0:2000], filtered_trace1)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(235)
    plt.title('2º Intervalo filt')
    plt.plot(time[2000:4000], filtered_trace2)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(236)
    plt.title('3º Intervalo filt')
    plt.plot(time[4000:], filtered_trace3)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.tight_layout()

    plt.figure()
    
    plt.subplot(231)
    plt.title('1º Intervalo Bruto')
    plt.plot(freq1[mask1], fft_trace1[mask1])
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Amplitude')

    plt.subplot(232)
    plt.title('2º Intervalo Bruto')
    plt.plot(freq2[mask2], fft_trace2[mask2])
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Amplitude')

    plt.subplot(233)
    plt.title('3º Intervalo bruto')
    plt.plot(freq3[mask3], fft_trace3[mask3])
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Amplitude')

    plt.subplot(234)
    plt.title('1º Intervalo filt')
    plt.plot(freq1[mask1], fft_filtered_trace1[mask1])
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Amplitude')

    plt.subplot(235)
    plt.title('2º Intervalo filt')
    plt.plot(freq2[mask2], fft_filtered_trace2[mask2])
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Amplitude')

    plt.subplot(236)
    plt.title('3º Intervalo filt')
    plt.plot(freq3[mask3], fft_filtered_trace3[mask3])
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Amplitude')

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

