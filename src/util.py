""" Random utility functions """
from functools import wraps
import numpy as np
import pandas as pd

# Things that get exposed from * import
__all__ = [
    "constantly", "complement", "identity", "thread",
    "execute_fn_with_args_and_or_kwargs", "toy_df",
    "insert_na",
    ]

def thread(arg, *fns):
    if len(fns) > 0:
        return thread(fns[0](arg), *fns[1:])
    else:
        return arg

def identity(x):
    return x

def constantly(x):
    """ Returns a function that takes any args and returns x """
    def func(*args, **kwargs):
        return x
    return func

def complement(fn):
    """ Return fn that outputs the opposite truth values of the
    input function
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return not fn(*args, **kwargs)
    return wrapper

def execute_fn_with_args_and_or_kwargs(fn, args, kwargs):
    """ If args + kwargs aren't accepted only args are passed in"""
    try:
        return fn(*args, **kwargs)
    except TypeError:
        return fn(*args)

def toy_df(n_rows=20, n_cols=5, missingness=0.2, min_val=0, max_val=1,
              missing_value=np.nan, rand_seed=1234, sample_prefix=None):
    """Generate an array or DataFrame with NaNs"""
    np.random.seed(rand_seed)
    X = np.random.uniform(
        low = min_val, high = max_val, size = n_rows * n_cols).reshape(n_rows, n_cols).astype(
        float)
    # check missingness
    if missingness > 0:
        # If missingness >= 1 then use it as approximate (see below) count
        if missingness >= 1:
            n_missing = int(missingness)
        else:
            n_missing = int(missingness * n_rows * n_cols)
            print(n_missing)
    
    # Introduce NaNs until n_miss "NAs" are inserted.
    missing_count = 0
    for i,j in zip(np.random.choice(n_rows, n_missing), np.random.choice(n_cols, n_missing)):
        if np.isnan(X[i][j]):
            continue
        else:
            X[i][j] = missing_value
            missing_count += 1
        if missing_count >= n_missing:
            break

    # check sample_prefix
    if sample_prefix is None:
        return X
    else:
        colNames = [sample_prefix + '_' + str(i) for i in range(0, n_cols)]
        return pd.DataFrame(X, columns=colNames)
    

def insert_na(df, n_miss, seed):
    np.random.seed(seed)
    nrow,ncol = df.shape
    na_count = 0
    if n_miss >= nrow*ncol:
        out_df = df.replace(df.values, np.nan)
    else:
        tmp = df.to_numpy()
        while(1):
            if na_count >= n_miss:
                break
            x_ind = np.random.choice(nrow)
            y_ind = np.random.choice(ncol)
            if not np.isnan(tmp[x_ind][y_ind]):
                tmp[x_ind][y_ind] = np.nan
                na_count += 1
        out_df = pd.DataFrame(tmp, index=df.index, columns=df.columns)
    return out_df