import numpy as np

def to_string(val, dig=5):
    """ For rounding error measures when included in plot titles """
    # val = np.round(val, dig)
    # s = str(val)
    if val < 0.01:
        s = np.format_float_scientific(val, precision=dig)
    else:
        s = np.format_float_positional(val, precision=dig)
    return(s)

