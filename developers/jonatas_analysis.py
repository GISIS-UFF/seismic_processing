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

def time_variant_filtering2(data, dt, start_normalization, num_windows):
    """
    Apply time-variant filtering to normalize the amplitude of a signal in specific time windows.
    The upper limit uses np.max(data), while the lower limit uses np.min(data) for normalization.

    Parameters:
    - data (numpy.ndarray): 1D array representing the signal.
    - dt (float): Time sampling interval.
    - normalization_start_time (float): Time (in seconds) to start normalization.
    - num_windows (int): Number of windows to divide the signal for normalization.

    Returns:
    - numpy.ndarray: The scaled signal after normalization.
    """
    normalization_start_time = int(normalization_start_time / dt)
    if normalization_start_time >= len(data):
        raise ValueError("normalization_start_time exceeds the length of the signal.")
    if num_windows <= 0:
        raise ValueError("num_windows must be greater than 0.")

    window_size = (len(data) - normalization_start_time) // num_windows
    max_scale = np.max(data)  # Use max(data) for the upper limit
    min_scale = np.min(data)  # Use min(data) for the lower limit

    signal_scaled = np.copy(data)
    for i in range(num_windows):
        start_idx = normalization_start_time + i * window_size
        end_idx = start_idx + window_size if i < num_windows - 1 else len(data)
        signal_max = np.max(np.abs(data[start_idx:end_idx]))  # Absolute maximum in the window

        if signal_max > 0:  # Avoid division by zero
            # Scale positive and negative parts separately
            normalized_segment = data[start_idx:end_idx] / signal_max
            normalized_segment = np.where(
                normalized_segment > 0,
                normalized_segment * max_scale,  # Scale positive values with max_scale
                normalized_segment * abs(min_scale)  # Scale negative values with min_scale
            )
            signal_scaled[start_idx:end_idx] = normalized_segment

    return signal_scaled


if __name__ == "__main__":
    
    # Parameters
    nt = 6001
    nx = 321
    dt = 0.001
    num_windows = 10
    normalization_start_time = 1.2  # Time in seconds

    data = np.reshape(np.fromfile('../data/elastic_iso_data_nStations321_nSamples6001_shot_6.bin', dtype=np.float32), (nx, nt))

    # Example 1D
    trace = data[50, :]
    time = np.arange(nt) * dt

    
    time_variant_trace = time_variant_filtering2(trace, dt, normalization_start_time, num_windows)

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(time, trace, label='Original Trace')
    plt.title('Original Trace')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plt.plot(time, time_variant_trace, label='Scaled Trace')
    plt.title('Scaled Trace')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # FFT
    freq = np.fft.fftfreq(nt, dt)
    mask = freq > 0 
    fft_trace = np.abs(np.fft.fft(trace))
    fft_time_variant_trace = np.abs(np.fft.fft(time_variant_trace))

    # ------------------------------------------------------------------------------------------------------------------------
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
    #plt.xlim(0,150)

    plt.subplot(223)
    plt.plot(time, time_variant_trace, label='Filtered Trace')
    plt.title('Filtered Trace')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(224)
    plt.plot(freq[mask], fft_time_variant_trace[mask], label='Filtered Trace FFT')
    plt.title('FFT do Sinal Filtrado')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    #plt.xlim(0,150)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------

    # Example seismogram
    data_filt = np.zeros_like(data)

    for i in range(len(data[:,0])):
            data_filt[i,:] = time_variant_filtering2(data[i,:], dt, normalization_start_time, num_windows)

    scale = np.percentile(data, 99)
    scale_filt = np.percentile(data_filt, 99)

    plt.figure(figsize=(10,7))

    plt.subplot(121)
    plt.imshow(data.T, aspect='auto', cmap='Grays', vmin=-scale, vmax=scale)
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(data_filt.T, aspect='auto', cmap='Grays', vmin=-scale_filt, vmax=scale_filt)
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    exit()

    # Aplicando o filtro no seismogram
    # data_filt = np.zeros_like(data)

    # for i in range(len(data[:,0])):
    #         data_filt[i,:] = butter_bandpass_filter(data[i,:], lowcut= 2, highcut=300, fs=fs)


    # data_filt2 = np.zeros_like(data)

    # for i in range(len(data[:,0])):
    #         data_filt2[i,:] = time_variant_filtering(data[i,:], lowcut_freqs, highcut_freqs, durations, fs)
    
 

    # time_variant_filtering
    
    # scale = np.percentile(data, 99)
    # scale_filt = np.percentile(data_filt, 99)
    # scale_filt2 = np.percentile(data_filt2, 99)

    # plt.figure(figsize=(10,7))

    # plt.subplot(231)
    # plt.imshow(data.T, aspect='auto', cmap='Grays', vmin=-scale, vmax=scale)
    # plt.colorbar()

    # plt.subplot(232)
    # plt.imshow(data_filt.T, aspect='auto', cmap='Grays', vmin=-scale_filt, vmax=scale_filt)
    # plt.colorbar()

    # plt.subplot(233)
    # plt.imshow(data_filt2.T, aspect='auto', cmap='Grays', vmin=-scale_filt2, vmax=scale_filt2)
    # plt.colorbar()

    # # Aplicando o ganho
    # data[0:50, 1200:6000] = data_filt2[0:50, 1200:6000] * 100
    # data_filt[0:50, 1200:6000] = data_filt2[0:50, 1200:6000] * 100
    # data_filt2[0:50, 1200:6000] = data_filt2[0:50, 1200:6000] * 1000

    # scale = np.percentile(data, 99)
    # scale_filt = np.percentile(data_filt, 99)
    # scale_filt2 = np.percentile(data_filt2, 99)

    # plt.subplot(234)
    # plt.imshow(data.T, aspect='auto', cmap='Grays', vmin=-scale, vmax=scale)
    # plt.colorbar()

    # plt.subplot(235)
    # plt.imshow(data_filt.T, aspect='auto', cmap='Grays', vmin=-scale_filt, vmax=scale_filt)
    # plt.colorbar()

    # plt.subplot(236)
    # plt.imshow(data_filt2.T, aspect='auto', cmap='Grays', vmin=-scale_filt2, vmax=scale_filt2)
    # plt.colorbar()

    # plt.tight_layout()
    # plt.show()

    # Minimum_cutoff_frequencies = [1, 2]  # em Hz
    # n_windows = 101
    # lowcut_freqs = np.linspace(2, 1, n_windows) # Variation of minimum frequencies (Hz) for each interval
    # highcut_freqs = np.linspace(29, 30, n_windows)  # Variation of Maximum frequencies (Hz) for each interval
    
    # # Duration of intervals (in seconds)
    # durations = np.zeros((n_windows)) + (nt * dt)/ n_windows

    # Time_variant_filtering
    # filtered_trace = time_variant_filtering(trace, lowcut_freqs, highcut_freqs, durations, fs)

    # Plot
    # plt.figure(figsize=(12, 7))

    # plt.subplot(121)
    # plt.plot(time, trace, label=' Original Trace')
    # plt.title('Original Trace')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.grid(True)

    # plt.subplot(122)
    # # plt.plot(time, trace_end, label='Filtered Trace')
    # plt.plot(time, signal_scaled, label='Filtered Trace')

    # plt.title('Filtered Trace')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.grid(True)

    # plt.show()




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

