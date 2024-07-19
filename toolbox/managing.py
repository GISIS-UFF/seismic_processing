import numpy as np
import segyio as sgy

def import_sgy_file(filename: str) -> sgy.SegyFile:

    try:
        return sgy.open(filename, ignore_geometry=True, mode="r+")
    except FileNotFoundError:
        print("File does not exist. Please verify the file path.")
        exit()

def export_sgy_file(data : sgy.SegyFile, filename : str) -> None:
    # Anthony
    dt = data.attributes(117)[0][0] 

    # Criar um novo arquivo SEG-Y
    spec = sgy.spec()
    spec.samples = data.samples
    spec.tracecount = data.tracecount
    spec.format = data.format

    
    with sgy.create(filename, spec) as f:
        
        for i in range(data.tracecount):
            f.trace[i] = data.trace[i]
        
        
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

def extract_trace_gather(data : sgy.SegyFile,filename : str,removetrace: list):
    dt = data.attributes(117)[0][0] 
    t=data.tracecount - len(removetrace)
    # Criar um novo arquivo SEG-Y
    spec = sgy.spec()
    spec.samples = data.samples
    spec.tracecount = t
    spec.format = data.format

    #apagar o traço 
    with sgy.create(filename, spec) as f:
        
        count=0
        for i in range(data.tracecount):
            if i not in removetrace:
                f.trace[count] = data.trace[i]
                f.header[count] = data.header[i]
                f.header[count][sgy.TraceField.TRACE_SAMPLE_INTERVAL] = dt
                count+=1
        f.bin = data.bin
        
        
       
        
       

def edit_header_attribute(data : sgy.SegyFile, keyword : int, attribute : np.ndarray) -> sgy.SegyFile:
    # Davi
    # bora ver se vai dar tempo para fazer na sexta feira depois do campo do Francisco
    current_data_header = np.where(data.attributes(keyword)[:] == attribute)[0]
    pass

