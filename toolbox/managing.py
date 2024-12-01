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
    
    all_indexes = get_keyword_indexes(data, key)

    index = [index] if not isinstance(index, (np.ndarray, list)) else index 

    for ind in index:
        if ind not in all_indexes:
            print(f"\033[31mIndex {ind} for key {key} is invalid!\033[m\
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

def edit_trace_header(data : sgy.SegyFile, bytes : list, values : list[np.ndarray]):
    '''
    Seismic data header editing.

    ### Parameters:

    data: segyio object.

    bytes: a sequence of bytes to edit.

    values: arrays with comlete values per trace corresponting with its bytes.

    ### Examples:

    Check examples folder for realistic trace edition.
    
    >>> mng.edit_trace_header(data, bytes = 1, values = [i+1 for i in range(nTraces)])    
    '''

    bytes = [bytes] if not isinstance(bytes, (np.ndarray, list)) else bytes 

    if len(bytes) != len(values):
        print("You must to assign each byte with its respective value array!")
        exit()

    for i in range(data.tracecount):
        for k, byte in enumerate(bytes):
            if byte in __keywords.values():
                data.header[i][byte] = int(values[k][i])
