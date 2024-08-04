from sys import path
path.append("../")

from toolbox import managing as mng
from toolbox import visualizing as view

# def semblance(data : sgy.SegyFile, index : int):
#     """
#     Plot the velocity semblance of the according CMP Gather

#     Parameters:
#     data: segyio object
#         The seismic data file.
#     index: int
#         The index of the CMP gather.
#     """
#     key = 'cmp'
#     __check_index(data, key, index)

#     byte, label = __keywords.get(key) # Obtém byte e label da keyword
#     traces = np.where(data.attributes(byte)[:] == index)[0]  # Seleciona os traços correspondentes ao índice de CMP
#     seismic = data.trace.raw[:].T  # Carrega os dados e transpõe a matriz
#     seismic = seismic[:, traces]  # Seleciona apenas os traços correspondentes ao índice

#     vmin = 0  # Velocidade mínima (m/s) - colocar editável com kwargs.get 
#     vmax = 6000  # Velocidade máxima (m/s) - colocar editável com kwargs.get
#     dv = 200  # Step da velocidade (m/s)
#     tmin = 0.0  # Tempo mínimo (s)
#     dt = 0.004  # Step de tempo (s)

#     nt = data.samples.size  # Número de amostras no tempo.
#     tmax = nt * dt  # Tempo máximo calculado com base no número de amostras e no incremento de tempo.

#     num_samples = int((tmax - tmin) / dt)  # Calcula o número de amostras de tempo.
#     num_velocities = int((vmax - vmin) / dv)  # Calcula o número de incrementos de velocidade.
#     semblance = np.zeros((num_samples, num_velocities))  # Inicializa a matriz de semblance.

#     time = np.arange(tmin, tmax, dt) # array do tempo
#     velocities = np.arange(vmin, vmax, dv) # array de velocidade

#     for 