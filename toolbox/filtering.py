import scipy as sc
import numpy as np
import segyio as sgy

def fourier_FX_domain(input : sgy.SegyFile, fmin : float, fmax : float, output_name : str) -> sgy.SegyFile:

    order = 5.0

    dt = input.attributes(117)[0][0] * 1e-6

    seismic_input = input.trace.raw[:].T

    seismic_bandpass = np.zeros_like(seismic_input)

    for i in range(input.tracecount):

        b, a = sc.signal.butter(order, [fmin, fmax], fs = 1/dt, btype = 'band')

        seismic_bandpass[:,i] = sc.signal.lfilter(b, a, seismic_input[:,i])

    sgy.tools.from_array2D(output_name, seismic_bandpass.T)
    
    output = sgy.open(output_name, "r+", ignore_geometry = True)
    output.header = input.header
    
    return output

def fourier_FK_domain(data : sgy.SegyFile, angle : float) -> sgy.SegyFile:
    
    pass