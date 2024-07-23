def semblance(data : sgy.SegyFile, index : int):
    """
    Plot the velocity semblance of the according CMP Gather


    """
    key = 'cmp'
    __check_index(data, key, index)

    byte, label = __keywords.get(key) # Obtém byte e label da keyword
    traces = np.where(data.attributes(byte)[:] == index)[0]  # Seleciona os traços correspondentes ao índice de CMP
    seismic = data.trace.raw[:].T  # Carrega os dados e transpõe a matriz
    seismic = seismic[:, traces]  # Seleciona apenas os traços correspondentes ao índice

    vmin = 0  # Velocidade mínima (m/s)
    vmax = 6000  # Velocidade máxima (m/s)
    dv = 200  # Step da velocidade (m/s)
    tmin = 0.0  # Tempo mínimo (s)
    dt = 0.004  # Step de tempo (s)