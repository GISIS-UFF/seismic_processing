import numpy as np
import segyio as sgy

__keywords = {'tsl'  : 1,   # trace sequence line
              'tsf'  : 5,   # trace sequence file     
              'src'  : 9,   # shot id 
              'rec'  : 13,  # receiver id 
              'cmp'  : 21,  # mid point id
              'off'  : 37,  # offset id 
              'zrec' : 41,  # receiver elevation
              'zsrc' : 45,  # source elevation
              'gscal': 69,  # geometry scalar
              'xsrc' : 73,  # source x
              'ysrc' : 77,  # source y
              'xrec' : 81,  # receiver x
              'yrec' : 85,  # receiver y
              'nt'   : 115, # time samples 
              'dt'   : 117, # time spacing 
              'xcmp' : 181, # mid point x
              'ycmp' : 185} # mid point y

def __check_keyword(key : str) -> None:
    
    if key not in __keywords.keys():
        print("\033[31mInvalid keyword!\033[m\
                     \nPlease use a valid header keyword: ['src', 'rec', 'off', 'cmp']")
        exit()

def __check_index(data : sgy.SegyFile, key : str, index : int ) -> None:   
    
    if index not in get_keyword_indexes(data, key):
        print("\033[31mInvalid index!\033[m\
                     \nPlease use the function \033[33mmng.get_keyword_indexes\033[m to choose a properly index.")
        exit()

def get_keyword_indexes(data : sgy.SegyFile, key : str) -> np.ndarray:
    '''
    Return possible indexes to access in seismic gather.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    ### Returns:

    An array with all possible indexes for indicated keyword.     

    ### Examples:
    
    >>> get_keyword_indexes(data, key = "src")
    >>> get_keyword_indexes(data, key = "rec")
    >>> get_keyword_indexes(data, key = "cmp")
    >>> get_keyword_indexes(data, key = "off")
    '''    

    __check_keyword(key)

    byte = __keywords.get(key)

    return np.unique(data.attributes(byte))

def get_full_fold_cmps(data : sgy.SegyFile) -> np.ndarray:
    '''
    Return full fold cmp indexes to access in seismic gather.
    
    ### Parameters:        
    
    data: segyio object.
    
    ### Returns:

    An array with all possible full fuld cmp indexes.     

    ### Examples:
    
    >>> get_full_fold_cmps(data)
    '''    
    
    key = "cmp"
    byte = __keywords.get(key)

    cmp_indexes = get_keyword_indexes(data, key)
    _, cmps_per_traces = np.unique(data.attributes(byte)[:], return_counts = True)
    complete_cmp_indexes = np.where(cmps_per_traces == np.max(cmps_per_traces))[0]
    indexes = cmp_indexes[complete_cmp_indexes[:]]

    return indexes

def import_sgy_file(input_name: str) -> sgy.SegyFile:
    '''
    Import seismic data in .segy format. 
    
    ### Parameters:        
    
    input_name: file path.
    
    ### Returns:

    A segy file object.
    '''

    try:
        return sgy.open(input_name, ignore_geometry = True, mode = "r+")
    except FileNotFoundError:
        print(f"File \033[31m{input_name}\033[m does not exist. Please verify the file path.")
        exit()

def show_trace_header(data : sgy.SegyFile) -> None:    

    traceHeader = sgy.tracefield.keys
    print(f"{'Trace header':>40s} {'byte':^6s} {'first':^11s} {'last':^11s} \n")
    
    for k, v in traceHeader.items():
        if v in __keywords.values():    
            first = data.attributes(v)[0][0]
            last = data.attributes(v)[data.tracecount-1][0]
            print(f"{k:>40s} {str(v):^6s} {str(first):^11s} {str(last):^11s}")

def gather_windowing(data : sgy.SegyFile, output_name : str, **kwargs) -> sgy.SegyFile:
    '''
    Windowing data according with keywords and time slicing.

    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    time_beg: initial time in seconds.

    time_end: final time in seconds.

    indexes_cut: a list of indexes to extract from seismic data. 
    
    ### Examples:

    >>> mng.extract_gather_window(data, output_name, key = "src", indexes_cut = [281, 282, 286])
    >>> mng.extract_gather_window(data, output_name, key = "cmp", indexes_cut = [631])
    '''    

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    key = kwargs.get("key") if "key" in kwargs else "src"

    byte = __keywords.get(key)

    all_indexes = get_keyword_indexes(data, key)

    time_beg = kwargs.get("time_beg") if "time_beg" in kwargs else 0.0
    time_end = kwargs.get("time_end") if "time_end" in kwargs else (nt-1)*dt

    if time_beg >= time_end:
        if time_beg >= 0.0 and time_end < (nt-1)*dt:
            print("Error: incorrect time for slicing!")
            
    indexes_cut = kwargs.get("indexes_cut") if "indexes_cut" in kwargs else all_indexes
    
    nTimes = int(int(time_end/dt) - (time_beg/dt)) + 1

    nTraces = np.zeros(len(indexes_cut), dtype = int)
    for index in range(len(indexes_cut)):
        nTraces[index] = len(np.where(data.attributes(byte)[:] == indexes_cut[index])[0])

    seismic = data.trace.raw[:].T
    
    seismic = seismic[int(time_beg/dt):int(time_end/dt)+1, :] 

    new_seismic = np.zeros((nTimes, np.sum(nTraces, dtype = int)), dtype = np.float32)

    for index in range(len(indexes_cut)):
        
        filling = slice(np.sum(nTraces[:index], dtype=int), np.sum(nTraces[:index+1], dtype=int))    
        picking = np.where(data.attributes(byte)[:] == indexes_cut[index])[0]

        new_seismic[:, filling] = seismic[:, picking]

    sgy.tools.from_array2D(output_name, new_seismic.T)
    output = sgy.open(output_name, "r+", ignore_geometry = True)

    header_values = np.fromiter(sgy.tracefield.keys.values(), dtype = int)

    for i in range(len(indexes_cut)):
        
        picking = np.where(data.attributes(byte)[:] == indexes_cut[index])[0]  
        
        skip = np.sum(nTraces[:i], dtype = int)
        
        for k in range(nTraces[i]):
            for w in header_values:    
                if w in __keywords.values():
                    output.header[skip + k][w] = data.attributes(w)[picking[k]][0] 

            output.header[skip + k][sgy.TraceField.TRACE_SAMPLE_COUNT] = nTimes
            output.header[skip + k][sgy.TraceField.TRACE_SAMPLE_INTERVAL] = int(dt * 1e6)

    return output

def edit_trace_header(data : sgy.SegyFile, bytes : list, values : list[np.ndarray]):
    '''
    Instructions


    '''

    if len(bytes) != len(values):
        print("Error message")

    for i in range(data.tracecount):
        for k, byte in enumerate(bytes):
            if byte in __keywords.values():
                data.header[i][byte] = int(values[k][i])
