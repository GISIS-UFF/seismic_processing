import numpy as np
import segyio as sgy

__keywords = {'src' : [9,  'shot'], 
              'rec' : [13, 'receiver'], 
              'off' : [37, 'offset'], 
              'cmp' : [21, 'mid point']}

def __check_keyword(key : str) -> None:
    
    if key not in __keywords.keys():
        print("\033[31mInvalid keyword!\033[m\
                     \nPlease use a valid header keyword: ['src', 'rec', 'off', 'cmp']")
        exit()

def __check_index(data : sgy.SegyFile, key : str, index : int ) -> None:   
    
    if index not in keyword_indexes(data, key):
        print("\033[31mInvalid index choice!\033[m\
                     \nPlease use the function \033[33mview.keyword_indexes\033[m to choose a properly index.")
        exit()

def keyword_indexes(data : sgy.SegyFile, key : str) -> np.ndarray:
    '''
    Print possible indexes to access in seismic gather.
    
    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    ### Examples:
    
    >>> keyword_indexes(data, key = "src")
    >>> keyword_indexes(data, key = "rec")
    >>> keyword_indexes(data, key = "cmp")
    >>> keyword_indexes(data, key = "off")
    '''    

    __check_keyword(key)

    byte = __keywords.get(key)[0]

    return np.unique(data.attributes(byte))

def import_sgy_file(input_name: str) -> sgy.SegyFile:
    '''
    Import seismic data in .segy format. 
    
    ### Parameters:        
    
    input_name: file path.
    '''

    try:
        return sgy.open(input_name, ignore_geometry=True, mode="r+")
    except FileNotFoundError:
        print(f"File \033[31m{input_name}\033[m does not exist. Please verify the file path.")
        exit()

def export_sgy_file(data : sgy.SegyFile, output_name : str) -> None:
    
    
    dt = data.attributes(117)[0][0] 

    # Criar um novo arquivo SEG-Y
    spec = sgy.spec()
    spec.samples = data.samples
    spec.tracecount = data.tracecount
    spec.format = data.format

    
    with sgy.create(output_name, spec) as f:
        
        for i in range(data.tracecount):
            f.trace[i] = data.trace[i]
        
        # Why are there two for loops here? (computationally expensive)
        f.bin = data.bin
        
       
        for i in range(data.tracecount):
            f.header[i] = data.header[i]
            f.header[i][sgy.TraceField.TRACE_SAMPLE_INTERVAL] = dt

def show_binary_header(data : sgy.SegyFile) -> None:

    binHeader = sgy.binfield.keys
    print("\n Checking binary header \n")
    print(f"{'key':>25s} {'byte':^6s} {'value':^7s} \n")
    for k, v in binHeader.items():
        if v in data.bin:
            print(f"{k:>25s} {str(v):^6s} {str(data.bin[v]):^7s}")

def show_trace_header(data : sgy.SegyFile) -> None:    

    traceHeader = sgy.tracefield.keys
    print("\n Checking trace header \n")
    print(f"{'Trace header':>40s} {'byte':^6s} {'first':^11s} {'last':^11s} \n")
    for k, v in traceHeader.items():
        first = data.attributes(v)[0][0]
        last = data.attributes(v)[data.tracecount-1][0]
        print(f"{k:>40s} {str(v):^6s} {str(first):^11s} {str(last):^11s}")

def gather_windowing(data : sgy.SegyFile, output_name : str, **kwargs):
    '''
    Windowing data according with keywords and time slicing.

    ### Parameters:        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    time_beg: initial time in seconds.

    time_end: final time in seconds.

    index_beg: initial index for specific keyword. 
    
    index_end: final index for specific keyword.

    ### Hint:

    If no keys or times provided, nothing going to happen.  
        
    ### Examples:

    >>> mng.extract_gather_window(data, output_name)
    >>> mng.extract_gather_window(data, output_name, key = "src", time_beg = 0.1, time_end = 0.5)
    >>> mng.extract_gather_window(data, output_name, key = "src", index_beg = 231, index_end = 231)
    '''    

    nt = data.attributes(115)[0][0]
    dt = data.attributes(117)[0][0] * 1e-6

    key = kwargs.get("key") if "key" in kwargs else "src"

    indexes = keyword_indexes(data, key)

    time_beg = kwargs.get("time_beg") if "time_beg" in kwargs else 0.0
    time_end = kwargs.get("time_end") if "time_end" in kwargs else (nt-1)*dt

    if time_beg >= time_end:
        if time_beg >= 0.0 and time_end < (nt-1)*dt:
            print("Error: incorrect time for slicing!")
            
    index_beg = kwargs.get("index_beg") if "index_beg" in kwargs else indexes[0]    
    index_end = kwargs.get("index_beg") if "index_beg" in kwargs else indexes[-1]   

    __check_index(data, key, index_beg)
    __check_index(data, key, index_end)

    limit_beg = np.where(indexes == index_beg)[0][0]
    limit_end = np.where(indexes == index_end)[0][0]

    byte,_ = __keywords.get(key)

    seismic = data.trace.raw[:].T
    
    seismic = seismic[int(time_beg/dt):int(time_end/dt) + 1, :] 

    nGathers = limit_end - limit_beg + 1
    nTraces = len(np.where(data.attributes(byte)[:] == index_end)[0])
    nTimes = int(int(time_end/dt) - (time_beg/dt)) + 1

    new_seismic = np.zeros((nTimes, nTraces*nGathers), dtype = np.float32)

    for k in range(limit_beg, limit_end + 1):
        
        filling = slice(k*nTraces, k*nTraces + nTraces)    
        picking = np.where(data.attributes(byte)[:] == indexes[k])[0]

        new_seismic[:, filling] = seismic[:, picking]

    sgy.tools.from_array2D(output_name, new_seismic.T)
    output = sgy.open(output_name, "r+", ignore_geometry = True)

    header_values = np.fromiter(sgy.tracefield.keys.values(), dtype = int)

    for i in range(limit_beg, limit_end + 1):
        
        picking = np.where(data.attributes(byte)[:] == indexes[i])[0]  

        for k in range(nTraces):

            for w in header_values:    
                output.header[i*nTraces + k][w] = data.attributes(w)[picking[k]][0] 

            output.header[i*nTraces + k][sgy.TraceField.TRACE_SAMPLE_COUNT] = nTimes

    return output

def mute_traces(data : sgy.SegyFile, output_name : str, trace_list : list):
    '''
    Erase amplitude of traces selected according with the header keyword TRACE_SEQUENCE_LINE.

    ### Parameters:        
    
    data: segyio object.

    trace_list: traces to be muted. 
        
    ### Examples:

    >>> mng.mute_traces(data, trace_list)
    '''    

    seismic = data.trace.raw[:].T
    
    traces = np.array(trace_list) - 1
    
    seismic[:, traces] = 1e-6

    sgy.tools.from_array2D(output_name, seismic.T)
    
    output = sgy.open(output_name, "r+", ignore_geometry = True)
    output.header = data.header

    return output

def edit_header_attribute(data : sgy.SegyFile, keyword : int, attribute : np.ndarray) -> sgy.SegyFile:
    # Davi
    # bora ver se vai dar tempo para fazer na sexta feira depois do campo do Francisco
    current_data_header = np.where(data.attributes(keyword)[:] == attribute)[0]
    pass

