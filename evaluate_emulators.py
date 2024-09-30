import numpy as np

def evaluate_emulators(x_raw, EMULATOR_FUNCTIONS, yscale_obj):
    """
    Return scaled emulator values
    """
    em_vals = np.zeros([x_raw.shape[0], 0])

    for emulator in EMULATOR_FUNCTIONS:
        y_em_raw = emulator(x_raw)
        y_em = yscale_obj.transform(y_em_raw.reshape(-1, 1))
        em_vals = np.hstack([em_vals, y_em])
    return(em_vals)

