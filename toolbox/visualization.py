import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

keywords = {'src' : [9,  'shot'], 
            'rec' : [13, 'receiver'], 
            'off' : [37, 'offset'], 
            'cmp' : [21, 'common mid point']}

def check_keyword(key : str) -> None:
    '''
    Documentation
    

    '''    
    
    if key not in keywords.keys():
        print("Invalid keyword!")
        print("Please use a valid header keyword: ['src', 'rec', 'off', 'cmp']")
        exit()

def keyword_indexes(data : sgy.SegyFile, key : str) -> np.ndarray:
    '''
    Documentation
    

    '''    

    check_keyword(key)

    byte = keywords.get(key)[0]
    possibilities = np.unique(data.attributes(byte))

    return possibilities

def seismic(data : sgy.SegyFile, key : str, index : int) -> None:
    '''
    Plot a seismic gather according with a specific header keyword.
    
    Parameters
    ----------        
    
    data: segyio object.

    key: header keyword options -> ["src", "rec", "off", "cmp"]
    
    index: integer that select a common gather.  

    Examples
    --------
    >>> plot_seismic(data, key = "src", index = 51)
    >>> plot_seismic(data, key = "rec", index = 203)
    >>> plot_seismic(data, key = "cmp", index = 315)
    >>> plot_seismic(data, key = "off", index = 223750)
    '''    

    check_keyword(key)

    byte, label = keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    seismic = data.trace.raw[:].T
    seismic = seismic[:, traces]

    scale = 0.99*np.std(seismic)

    fig, ax = plt.subplots(num = f"Common {label} gather", ncols = 1, nrows = 1, figsize = (10, 5))

    ax.imshow(seismic, aspect = "auto", cmap = "Greys", vmin = -scale, vmax = scale)

    # define axis values according with key
    # define labels according with key
    # define colorbar correctly
        
    fig.tight_layout()
    plt.show()

def geometry(data : sgy.SegyFile, key : str, index : int) -> None:
    '''
    Documentation
    
    
    '''    

    check_keyword(key)

    byte, label = keywords.get(key)

    traces = np.where(data.attributes(byte)[:] == index)[0]

    sx = data.attributes(73)[traces] / data.attributes(71)[traces]
    sy = data.attributes(77)[traces] / data.attributes(71)[traces]    
    sz = data.attributes(45)[traces] / data.attributes(71)[traces]

    rx = data.attributes(81)[traces] / data.attributes(69)[traces]
    ry = data.attributes(85)[traces] / data.attributes(69)[traces]    
    rz = data.attributes(41)[traces] / data.attributes(69)[traces]

    cmpx = data.attributes(181)[traces] / data.attributes(69)[traces]
    cmpy = data.attributes(185)[traces] / data.attributes(69)[traces]    


    fig, ax = plt.subplots(num = f"Common {label} gather", ncols = 1, nrows = 1, figsize = (10, 5))

    # adjust order of plots according with key
    # example: 
    #    src key order -> 1 cmp, 2 receiver and 3 shot
    #    cmp key order -> 1 shot, 2 receiver and 3 cmp          

    ax.plot(cmpx, cmpy, 'ob') # 1
    ax.plot(rx, ry, 'oy')     # 2
    ax.plot(sx, sy, 'og')     # 3

    # to show depth with colors
    # ax.scatter(rx, rx, c = rz, cmap = ???)
    # ax.scatter(sx, sx, c = sz, cmap = ???)

    # define axis values according with key
    # define labels according with key
    # define colorbar correctly
        
    fig.tight_layout()
    plt.show()

    pass