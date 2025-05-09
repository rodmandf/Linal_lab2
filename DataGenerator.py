import numpy as np

def data_generator(m:int , n:int , seed:int = None):
    if seed != None: (
        np.random.seed(seed)
    )
    X = np.random.randn(m,n)
    w_true = np.random.randn(n)
    b_true = np.random.randn()
    z_true = X.dot(w_true) + b_true
    y = (z_true >= 0).astype(int)

    return X, y