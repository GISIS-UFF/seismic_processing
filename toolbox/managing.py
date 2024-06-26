import segyio as sgy

def import_sgy_file(filename : str) -> sgy.SegyFile:
    '''
    Documentation
    

    '''
    
    return sgy.open(filename, ignore_geometry = True)

def show_binary_header(data : sgy.SegyFile) -> None:
    '''
    Documentation


    '''
    
    binHeader = sgy.binfield.keys
    print("\n Checking binary header \n")
    print(f"{'key':>25s} {'byte':^6s} {'value':^7s} \n")
    for k, v in binHeader.items():
        if v in data.bin:
            print(f"{k:>25s} {str(v):^6s} {str(data.bin[v]):^7s}")

def show_trace_header(data : sgy.SegyFile) -> None:    
    '''
    Documentation


    '''

    traceHeader = sgy.tracefield.keys
    print("\n Checking trace header \n")
    print(f"{'Trace header':>40s} {'byte':^6s} {'first':^11s} {'last':^11s} \n")
    for k, v in traceHeader.items():
        first = data.attributes(v)[0][0]
        last = data.attributes(v)[data.tracecount-1][0]
        print(f"{k:>40s} {str(v):^6s} {str(first):^11s} {str(last):^11s}")
