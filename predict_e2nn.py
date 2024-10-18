import numpy as np

from evaluate_emulators import evaluate_emulators

def predict_e2nn(x_raw, model, EMULATOR_FUNCTIONS, xscale_obj, yscale_obj):
    """
    Predict the raw response of an e2nn model
    """
    # # good for small predictions
    # y = model([x, em])
    # # good for big predictions (I settled on 10_000 earlier, but it might 
    # # take longer than that to matter)
    # y = model.predict([x, em])

    x = xscale_obj.transform(x_raw)
    em = evaluate_emulators(x_raw, EMULATOR_FUNCTIONS, yscale_obj)

    # ideally would use MAX_FLOATS and N_HIDDEN*N_PRED 
    # (the size of the largest matrix of neuron values)
    # and only then break up predictions to prevent running out of memory
    # (Using simple heuristic for now)
    MAX_POINTS = 1_000  
    n_pts = x.shape[0]
    if n_pts <= MAX_POINTS:
        y = model([x, em])
    else:
        y = np.zeros([0,1])
        n_runs = np.ceil(n_pts/MAX_POINTS)
        x_frags = np.array_split(x, n_runs, axis=0)
        em_frags = np.array_split(em, n_runs, axis=0)
        for x_frag, em_frag in zip(x_frags, em_frags):
            y_frag = model([x_frag, em_frag])
            y = np.vstack([y, y_frag])

    y = yscale_obj.inverse_transform(y)
    y = y.flatten()

    return(y)

