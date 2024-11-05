""" Random utility functions """
import sys
from functools import wraps
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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
    """Insert a specified number of missing values into the DataFrame."""
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


def cluster_cols(data, k=2, random_state=0, n_init="auto"):
    """
    Binarize the columns of the DataFrame into two groups using K-means
    clustering (K=2). Note that the value of K is fixed and only supports
    two clusters.

    Missing values will be replaced with zero, while non-missing values will
    be replaced with one. K-means clustering (K=2) will then be applied to
    the columns to group samples with similar missing patterns.
    """
    #key is groupID, value is a list of samples (i.e., column names)
    group = {} 
    df = data.copy()
    names = df.columns
    #replace non missing values with 1
    df = df.mask(df.notna(), 1)
    #replace missing values with 0
    df = df.fillna(0)
    #transpose df, since we want to cluster columns
    df = df.T

    # Initialize KMeans with desired number of clusters (k)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(df)
    labels = [str(i) for i in kmeans.labels_]

    for i,j in zip(names, labels):
        if j not in group:
            group[j] = [i]
        else:
            group[j].append(i)
    return group


def calculate_metrics(df_true, df_pred, indices):
    """
    Calculates MAE, RAE, RMSE, and R-squared for given true and predicted values.
    Args:
        df_true : pd.DataFrame
                DataFrame of true values.
        df_pred : pd.DataFrame
                DataFrame of imputed values
        indices : List of list
                (x,y) location of missing values
    Returns:
        List
            MNAE (float): Mean Absolute Error.
            MDAE (float): Median Absolute Error.
            RAE (float): Relative Absolute Error
            RMSE (float): Root Mean Squared Error.
            MAPE (float): Mean Absolute Percentage Error
            R2 (float): R-squared score.
    """

    y_true = np.array([df_true.to_numpy()[i][j] for i,j in indices])
    y_pred = np.array([df_pred.to_numpy()[i][j] for i,j in indices])

    #absolute errors
    ab_errors = np.abs(y_true - y_pred)

    #mean absolute error
    MNAE = ab_errors.mean()
    MDAE = np.median(ab_errors)

    #relative absolute error
    RAE = ab_errors.sum()/np.sum(np.abs(y_true - np.mean(y_true)))

    #root mean square error
    RMSE = np.sqrt(np.square(ab_errors).mean())

    #Mean Absolute Percentage Error
    MAPE = np.mean(ab_errors/np.abs(y_true))

    R2 = np.corrcoef(y_true, y_pred)[0][1]
    return [float(i) for i in [MNAE, MDAE, RAE, MAPE, RMSE, R2]]

def calculate_errors(df_true, df_pred, indices):
    """
    Calculates errors (pred - true)
    Args:
        df_true : pd.DataFrame
                DataFrame of true values.
        df_pred : pd.DataFrame
                DataFrame of imputed values
        indices : List of list
                (x,y) location of missing values
    Returns:
        np.array
    """
    y_true = np.array([df_true.to_numpy()[i][j] for i,j in indices])
    y_pred = np.array([df_pred.to_numpy()[i][j] for i,j in indices])

    errors = y_pred - y_true
    return errors