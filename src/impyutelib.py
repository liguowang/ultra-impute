#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:01:45 2024
Adapted and modified from impyute.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.spatial import KDTree
from functools import wraps

## Common operations on matrices

def nan_indices(data):
    """ Finds the indices of all missing values.

    Parameters
    ----------
    data: numpy.ndarray

    Returns
    -------
    List of tuples
        Indices of all missing values in tuple format; (i, j)
    """
    return np.argwhere(np.isnan(data))

def map_nd(fn, arr):
    """ Map fn that takes a value over entire n-dim array

    Parameters
    ----------
    arr: numpy.ndarray

    Returns
    -------
    numpy.ndarray

    """
    return np.vectorize(fn)(arr)

def every_nd(fn, arr):
    """ Returns bool, true if fn is true for all elements of arr

    Parameters
    ----------
    arr: numpy.ndarray

    Returns
    -------
    bool

    """
    return all(map(fn, arr.flatten()))









## Util

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

def apply_method(df, method_name, **kwargs):
    """Applies a pandas method to a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to apply the method to.
        method_name (str): The name of the method to apply.
        **kwargs: Additional keyword arguments to pass to the method.
    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    method = getattr(df, method_name)
    return method(**kwargs)

def shepards(distances, power=2):
    """ Basic inverse distance weighting function

    Parameters
    ----------
    distances: list/numpy.ndarray
        1D list of numbers (ex. distance results from call to KDTree.query)

    power: int
        Default of 2 used since the referenced paper stated an exponent of 2 "gives seemingly
        satisfactory results"

    Returns
    -------
    numpy.ndarray
        1D list of numbers that sum to 1, represents weights of provided distances, in order.

    References
    ----------

    Shepard, Donald (1968). "A two-dimensional interpolation function for irregularly-spaced data".
    Proceedings of the 1968 ACM National Conference. pp. 517-524. doi:10.1145/800186.810616
    """
    return to_percentage(1/np.power(distances, power))

def to_percentage(vec):
    """ Converts list of real numbers into a list of percentages """
    return vec/np.sum(vec)




## Wrapper

def handle_df(fn):
    """ Decorator to handle pandas Dataframe object as input

    If the first arg is a pandas dataframe, convert it to a numpy array
    otherwise don't do anything. Cast back to a pandas Dataframe after
    the imputation function has run
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        is_df = False
        ## convert tuple to list so args can be modified
        args = list(args)
        ## Either make a copy or use a pointer to the original
        if kwargs.get('inplace'):
            args[0] = args[0]
        else:
            args[0] = args[0].copy()

        ## If input data is a dataframe then cast the input to an np.array
        ## and set an indicator flag before continuing
        if isinstance(args[0], pd.DataFrame):
            is_df = True
            in_ind = args[0].index
            in_columns = args[0].columns
            args[0] = args[0].to_numpy()

        ## function invokation
        results = execute_fn_with_args_and_or_kwargs(fn, args, kwargs)

        ## cast the output back to a DataFrame.
        if is_df:
            results = pd.DataFrame(results, index=in_ind, columns=in_columns)
        return results
    return wrapper

def add_inplace_option(fn):
    """ Decorator for inplace option

    Functions wrapped by this can have an `inplace` kwarg to use either a copy of
    data or reference """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        """ Run input checks"""
        ## convert tuple to list so args can be modified
        args = list(args)
        ## Either make a copy or use a pointer to the original
        if kwargs.get('inplace'):
            args[0] = args[0]
        else:
            args[0] = args[0].copy()

        ## function invokation
        return execute_fn_with_args_and_or_kwargs(fn, args, kwargs)
    return wrapper

def conform_output(fn):
    """ Decorator to handle impossible values

    Adds two optional kwargs, `coerce_fn` and `valid_fn`.

    `valid_fn` function stub

        def my_coerce_fn(some_literal) -> boolean

    `coerce_fn` function stub

        def my_coerce_fn(arr, x_i, y_i) -> some_literal

    Valid function is something run on each element of the, this is
    the function that we use to indicate whether the value is valid
    or not

    Coerce function has three arguments, the original matrix and
    the two indices of the invalid value x_i and y_i. This function
    will be run on all invalid values.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        def raise_error(arr, x_i, y_i):
            raise Exception("{} does not conform".format(arr[x_i, y_i]))
        ## convert tuple to list so args can be modified
        args = list(args)
        # function that checks if the value is valid
        valid_fn = kwargs.get("valid_fn", constantly(True))
        # function that modifies the invalid value to something valid
        coerce_fn = kwargs.get("coerce_fn", raise_error)

        ## function invokation
        results = execute_fn_with_args_and_or_kwargs(fn, args, kwargs)

        # check each value to see if it's valid
        bool_arr = map_nd(complement(valid_fn), results)
        # get indices of invalid values
        invalid_indices = np.argwhere(bool_arr)
        # run the coerce fn on each invalid indice
        for x_i, y_i in invalid_indices:
            results[x_i, y_i] = coerce_fn(results, x_i, y_i)

        return results
    return wrapper

def wrappers(fn):
    """ Helper decorator, all wrapper functions applied to modify input (matrix
    with missing values) and output (matrix with imputed values)

    NOTE: `handle_df` has to be last as it needs to be in the outer loop (first
    entry point) since every other function assumes you're getting an np.array
    as input
    """
    return thread(
        fn,                 # function that's getting wrapped
        add_inplace_option, # allow choosing reference/copy
        conform_output,     # allow enforcing of some spec on returned outputs
        handle_df,          # if df type, cast to np.array on in and df on out
    )

## Central tendency
@wrappers
def mean(data):
    """ Substitute missing values with the mean of that column.

    Parameters
    ----------
    data: numpy.ndarray
        Data to impute.

    Returns
    -------
    numpy.ndarray
        Imputed data.

    """
    nan_xy = nan_indices(data)
    for x_i, y_i in nan_xy:
        row_wo_nan = data[:, [y_i]][~np.isnan(data[:, [y_i]])]
        new_value = np.mean(row_wo_nan)
        data[x_i][y_i] = new_value
    return data

@wrappers
def median(data):
    """ Substitute missing values with the median of that column(middle).

    Parameters
    ----------
    data: numpy.ndarray
        Data to impute.

    Returns
    -------
    numpy.ndarray
        Imputed data.

    """
    nan_xy = nan_indices(data)
    cols_missing = set(nan_xy.T[1])
    medians = {}
    for y_i in cols_missing:
        cols_wo_nan = data[:, [y_i]][~np.isnan(data[:, [y_i]])]
        median_y = np.median(cols_wo_nan)
        medians[str(y_i)] = median_y
    for x_i, y_i in nan_xy:
        data[x_i][y_i] = medians[str(y_i)]
    return data

@wrappers
def mode(data):
    """ Substitute missing values with the mode of that column(most frequent).

    In the case that there is a tie (there are multiple, most frequent values)
    for a column randomly pick one of them.

    Parameters
    ----------
    data: numpy.ndarray
        Data to impute.

    Returns
    -------
    numpy.ndarray
        Imputed data.

    """
    nan_xy = nan_indices(data)
    modes = []
    for y_i in range(np.shape(data)[1]):
        unique_counts = np.unique(data[:, [y_i]], return_counts=True)
        max_count = np.max(unique_counts[1])
        mode_y = [unique for unique, count in np.transpose(unique_counts)
                  if count == max_count and not np.isnan(unique)]
        modes.append(mode_y)  # Appends index of column and column modes
    for x_i, y_i in nan_xy:
        data[x_i][y_i] = np.random.choice(modes[y_i])
    return data



#################
## random impute
#################
@wrappers
def random_impute(data):
    """ Fill missing values in with a randomly selected value from the same
    column.

    Parameters
    ----------
    data: numpy.ndarray
        Data to impute.

    Returns
    -------
    numpy.ndarray
        Imputed data.

    """
    nan_xy = nan_indices(data)
    for x, y in nan_xy:
        uniques = np.unique(data[:, y])
        uniques = uniques[~np.isnan(uniques)]
        data[x][y] = np.random.choice(uniques)
    return data


########################
## moving window impute
########################
@wrappers
def moving_window(data, nindex=None, wsize=5, errors="coerce", func=np.mean,
        inplace=False):
    """ Interpolate the missing values based on nearby values.

    For example, with an array like this:

        array([[-1.24940, -1.38673, -0.03214945,  0.08255145, -0.007415],
               [ 2.14662,  0.32758 , -0.82601414,  1.78124027,  0.873998],
               [-0.41400, -0.977629,         nan, -1.39255344,  1.680435],
               [ 0.40975,  1.067599,  0.29152388, -1.70160145, -0.565226],
               [-0.54592, -1.126187,  2.04004377,  0.16664863, -0.010677]])

    Using a `k` or window size of 3. The one missing value would be set
    to -1.18509122. The window operates on the horizontal axis.

    Usage
    -----

    The parameters default the function to a moving mean. You may want to change
    the default window size:

        moving_window(data, wsize=10)

    To only look at past data (null value is at the rightmost index in the window):

        moving_window(data, nindex=-1)

    To use a custom function:

        moving_window(data, func=np.median)

    You can also do something like take 1.5x the max of previous values in the window:

        moving_window(data, func=lambda arr: max(arr) * 1.50, nindex=-1)

    Parameters
    ----------
    data: numpy.ndarray
        2D matrix to impute.
    nindex: int
        Null index. Index of the null value inside the moving average window.
        Use cases: Say you wanted to make value skewed toward the left or right
        side. 0 would only take the average of values from the right and -1
        would only take the average of values from the left
    wsize: int
        Window size. Size of the moving average window/area of values being used
        for each local imputation. This number includes the missing value.
    errors: {"raise", "coerce", "ignore"}
        Errors will occur with the indexing of the windows - for example if there
        is a nan at data[x][0] and `nindex` is set to -1 or there is a nan at
        data[x][-1] and `nindex` is set to 0. `"raise"` will raise an error,
        `"coerce"` will try again using an nindex set to the middle and `"ignore"`
        will just leave it as a nan.
    inplace: {True, False}
        Whether to return a copy or run on the passed-in array

    Returns
    -------
    numpy.ndarray
        Imputed data.

    """
    if errors == "ignore":
        raise Exception("`errors` value `ignore` not implemented yet. Sorry!")

    if not inplace:
        data = data.copy()

    if nindex is None: # If using equal window side lengths
        assert wsize % 2 == 1, "The parameter `wsize` should not be even "\
        "if the value `nindex` is not set since it defaults to the midpoint "\
        "and an even `wsize` makes the midpoint ambiguous"
        wside_left = wsize // 2
        wside_right = wsize // 2
    else: # If using custom window side lengths
        assert nindex < wsize, "The null index must be smaller than the window size"
        if nindex == -1:
            wside_left = wsize - 1
            wside_right = 0
        else:
            wside_left = nindex
            wside_right = wsize - nindex - 1

    while True:
        nan_xy = nan_indices(data)
        n_nan_prev = len(nan_xy)
        for x_i, y_i in nan_xy:
            left_i = max(0, y_i-wside_left)
            right_i = min(len(data), y_i+wside_right+1)
            window = data[x_i, left_i: right_i]
            window_not_null = window[~np.isnan(window)]

            if len(window_not_null) > 0:
                try:
                    data[x_i][y_i] = func(window_not_null)
                    continue
                except Exception as e:
                    if errors == "raise":
                        raise e

            if errors == "coerce":
                # If either the window has a length of 0 or the aggregate function fails somehow,
                # do a fallback of just trying the best we can by using it as the middle and trying
                # to recalculate. Use temporary wside_left/wside_right, for only the calculation of
                # this specific problamatic value
                wside_left_tmp = wsize // 2
                wside_right_tmp = wside_left_tmp

                left_i_tmp = max(0, y_i-wside_left_tmp)
                right_i_tmp = min(len(data), y_i+wside_right_tmp+1)

                window = data[x_i, left_i_tmp:right_i_tmp]
                window_not_null = window[~np.isnan(window)]
                try:
                    data[x_i][y_i] = func(window_not_null)
                except Exception as e:
                    print("Exception:", e)
        if n_nan_prev == len(nan_indices(data)):
            break
    return data


########################
## fKNN
########################
@wrappers
def fKNN(data, na_locations, k=3, eps=0, p=2, distance_upper_bound=np.inf, leafsize=10,
        idw_fn=shepards, init_impute_fn=mean):
    """ Impute using a variant of the nearest neighbours approach

    Basic idea: Impute array with a passed in initial impute fn (mean impute)
    and then use the resulting complete array to construct a KDTree. Use this
    KDTree to compute nearest neighbours.  After finding `k` nearest
    neighbours, take the weighted average of them. Basically, find the nearest
    row in terms of distance

    This approach is much, much faster than the other implementation (fit+transform
    for each subset) which is almost prohibitively expensive.

    Parameters
    ----------
    data: ndarray
        2D matrix to impute.

    na_locations: tuple
        Pre-calculated (x,y) of missing values.

    k: int, optional
        Parameter used for method querying the KDTree class object. Number of
        neighbours used in the KNN query. Refer to the docs for
        [`scipy.spatial.KDTree.query`]
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html).

    eps: nonnegative float, optional
        Parameter used for method querying the KDTree class object. From the
        SciPy docs: "Return approximate nearest neighbors; the kth returned
        value is guaranteed to be no further than (1+eps) times the distance to
        the real kth nearest neighbor". Refer to the docs for
        [`scipy.spatial.KDTree.query`]
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html).

    p : float, 1<=p<=infinity, optional
        Parameter used for method querying the KDTree class object. Straight from the
        SciPy docs: "Which Minkowski p-norm to use. 1 is the
        sum-of-absolute-values Manhattan distance 2 is the usual Euclidean
        distance infinity is the maximum-coordinate-difference distance". Refer to
        the docs for
        [`scipy.spatial.KDTree.query`]
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html).

    distance_upper_bound : nonnegative float, optional
        Parameter used for method querying the KDTree class object. Straight
        from the SciPy docs: "Return only neighbors within this distance. This
        is used to prune tree searches, so if you are doing a series of
        nearest-neighbor queries, it may help to supply the distance to the
        nearest neighbor of the most recent point." Refer to the docs for
        [`scipy.spatial.KDTree.query`]
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html).

    leafsize: int, optional
        Parameter used for construction of the `KDTree` class object. Straight from
        the SciPy docs: "The number of points at which the algorithm switches
        over to brute-force. Has to be positive". Refer to the docs for
        [`scipy.spatial.KDTree`](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html)
        for more information.

    idw_fn: fn, optional
        Function that takes one argument, a list of distances, and returns weighted percentages. You can define a custom
        one or bootstrap from functions defined in `impy.util.inverse_distance_weighting` which can be using
        functools.partial, for example: `functools.partial(impy.util.inverse_distance_weighting.shepards, power=1)`

    init_impute_fn: fn, optional

    Returns
    -------
    numpy.ndarray
        Imputed data.

    Examples
    --------

        >>> data = np.arange(25).reshape((5, 5)).astype(np.float)
        >>> data[0][2] =  np.nan
        >>> data
        array([[ 0.,  1., nan,  3.,  4.],
               [ 5.,  6.,  7.,  8.,  9.],
               [10., 11., 12., 13., 14.],
               [15., 16., 17., 18., 19.],
               [20., 21., 22., 23., 24.]])
        >> fast_knn(data, k=1) # Weighted average (by distance) of nearest 1 neighbour
        array([[ 0.,  1.,  7.,  3.,  4.],
               [ 5.,  6.,  7.,  8.,  9.],
               [10., 11., 12., 13., 14.],
               [15., 16., 17., 18., 19.],
               [20., 21., 22., 23., 24.]])
        >> fast_knn(data, k=2) # Weighted average of nearest 2 neighbours
        array([[ 0.        ,  1.        , 10.08608891,  3.        ,  4.        ],
               [ 5.        ,  6.        ,  7.        ,  8.        ,  9.        ],
               [10.        , 11.        , 12.        , 13.        , 14.        ],
               [15.        , 16.        , 17.        , 18.        , 19.        ],
               [20.        , 21.        , 22.        , 23.        , 24.        ]])
        >> fast_knn(data, k=3)
        array([[ 0.        ,  1.        , 13.40249283,  3.        ,  4.        ],
               [ 5.        ,  6.        ,  7.        ,  8.        ,  9.        ],
               [10.        , 11.        , 12.        , 13.        , 14.        ],
               [15.        , 16.        , 17.        , 18.        , 19.        ],
               [20.        , 21.        , 22.        , 23.        , 24.        ]])
        >> fast_knn(data, k=5) # There are at most only 4 neighbours. Raises error
        ...
        IndexError: index 5 is out of bounds for axis 0 with size 5

    """
    nan_xy = na_locations #pre-calculate nan_xy
    data_c = data #pre-impute data
    kdtree = KDTree(data_c, leafsize=leafsize)

    for x_i, y_i in nan_xy:
        distances, indices = kdtree.query(data_c[x_i], k=k+1, eps=eps, p=p, 
                                          distance_upper_bound=distance_upper_bound)
        # Will always return itself in the first index. Delete it.
        distances, indices = distances[1:], indices[1:]
        # Add small constant to distances to avoid division by 0
        distances += 1e-3
        weights = idw_fn(distances)
        # Assign missing value the weighted average of `k` nearest neighbours
        data[x_i][y_i] = np.dot(weights, [data_c[ind][y_i] for ind in indices])
    return data


def external_ref(data, na_locations, ref_data, k=3, eps=0, p=2, 
                 distance_upper_bound=np.inf, leafsize=10,
                 idw_fn=shepards):
    """ Impute using a variant of the nearest neighbours approach

    Basic idea: Impute array with a passed in initial impute fn (mean impute)
    and then use the resulting complete array to construct a KDTree. Use this
    KDTree to compute nearest neighbours.  After finding `k` nearest
    neighbours, take the weighted average of them. Basically, find the nearest
    row in terms of distance

    This approach is much, much faster than the other implementation 
    (fit+transform for each subset) which is almost prohibitively expensive.

    Parameters
    ----------
    data: ndarray
        2D matrix with missing values.

    na_locations: tuple
        Pre-calculated (x,y) of missing values.
    
    ref_data: ndarray
        2D matrix used as external reference data. k nearest neighbours will be
        identified from this data.

    k: int, optional
        Parameter used for method querying the KDTree class object. Number of
        neighbours used in the KNN query.

    eps: nonnegative float, optional
        Parameter used for method querying the KDTree class object. From the
        SciPy docs: "Return approximate nearest neighbors; the kth returned
        value is guaranteed to be no further than (1+eps) times the distance to
        the real kth nearest neighbor".

    p : float, 1<=p<=infinity, optional
        Parameter used for method querying the KDTree class object. Straight
        from the SciPy docs: "Which Minkowski p-norm to use. 1 is the
        sum-of-absolute-values Manhattan distance 2 is the usual Euclidean
        distance infinity is the maximum-coordinate-difference distance".

    distance_upper_bound : nonnegative float, optional
        Parameter used for method querying the KDTree class object. Straight
        from the SciPy docs: "Return only neighbors within this distance. This
        is used to prune tree searches, so if you are doing a series of
        nearest-neighbor queries, it may help to supply the distance to the
        nearest neighbor of the most recent point." 

    leafsize: int, optional
        Parameter used for construction of the `KDTree` class object. Straight
        from the SciPy docs: "The number of points at which the algorithm
        switches over to brute-force. Has to be positive". 

    idw_fn: fn, optional
        Function that takes one argument, a list of distances, and returns 
        weighted percentages. You can define a custom one or bootstrap from
        functions defined in `impy.util.inverse_distance_weighting` which can
        be using functools.partial, for example:`functools.partial
        impy.util.inverse_distance_weighting.shepards, power=1)`

    Returns
    -------
    numpy.ndarray
        Imputed data.

    """
    nan_xy = na_locations #pre-calculate nan_xy
    kdtree = KDTree(ref_data, leafsize=leafsize)

    for x_i, y_i in nan_xy:
        distances, indices = kdtree.query(data[x_i], k=k, eps=eps, p=p, 
                                          distance_upper_bound=distance_upper_bound)
        # Add small constant to distances to avoid division by 0
        distances += 1e-3
        weights = idw_fn(distances)
        # Assign missing value the weighted average of `k` nearest neighbours
        data[x_i][y_i] = np.dot(weights, [ref_data[ind][y_i] for ind in indices])
    return data


############################
## Expectation–maximization
############################
@wrappers
def em(data, eps=0.1):
    """ Imputes given data using expectation maximization.

    E-step: Calculates the expected complete data log likelihood ratio.
    M-step: Finds the parameters that maximize the log likelihood of the
    complete data.

    Parameters
    ----------
    data: numpy.nd.array
        Data to impute.
    eps: float
        The amount of minimum change between iterations to break, if relative
        change < eps, converge.
        relative change = abs(current - previous) / previous
    inplace: boolean
        If True, operate on the numpy array reference

    Returns
    -------
    numpy.nd.array
        Imputed data.

    """
    nan_xy =  nan_indices(data)
    for x_i, y_i in nan_xy:
        col = data[:, int(y_i)]
        mu = col[~np.isnan(col)].mean()
        std = col[~np.isnan(col)].std()
        col[x_i] = np.random.normal(loc=mu, scale=std)
        previous, i = 1, 1
        while True:
            i += 1
            # Expectation
            mu = col[~np.isnan(col)].mean()
            std = col[~np.isnan(col)].std()
            # Maximization
            col[x_i] = np.random.normal(loc=mu, scale=std)
            # Break out of loop if likelihood doesn't change at least 10%
            # and has run at least 5 times
            delta = np.abs(col[x_i]-previous)/previous
            if i > 5 and delta < eps:
                data[x_i][y_i] = col[x_i]
                break
            data[x_i][y_i] = col[x_i]
            previous = col[x_i]
    return data


#######################
## Buck's method
#######################
@wrappers
def buck_iterative(data, eps=0.1):
    """ Iterative variant of buck's method

    - Variable to regress on is chosen at random.
    - EM type infinite regression loop stops after change in prediction from
      previous prediction < 10% for all columns with missing values

    A Method of Estimation of Missing Values in Multivariate Data Suitable for
    use with an Electronic Computer S. F. Buck Journal of the Royal Statistical
    Society. Series B (Methodological) Vol. 22, No. 2 (1960), pp. 302-306

    Parameters
    ----------
    data: numpy.ndarray
        Data to impute.
    eps: float
        The amount of minimum change between iterations to break, if relative 
        change < eps, converge.
        relative change = abs(current - previous) / previous
    Returns
    -------
    numpy.ndarray
        Imputed data.

    """
    nan_xy = nan_indices(data)

    # Add a column of zeros to the index values
    nan_xyz = np.append(nan_xy, np.zeros((np.shape(nan_xy)[0], 1)), axis=1)

    nan_xyz = [[int(x), int(y), v] for x, y, v in nan_xyz]
    temp = []
    cols_missing = {y for _, y, _ in nan_xyz}

    # Step 1: Simple Imputation, these are just placeholders
    for x_i, y_i, value in nan_xyz:
        # Column containing nan value without the nan value
        col = data[:, [y_i]][~np.isnan(data[:, [y_i]])]

        new_value = np.mean(col)
        data[x_i][y_i] = new_value
        temp.append([x_i, y_i, new_value])
    nan_xyz = temp

    # Step 5: Repeat step 2 - 4 until convergence (the 100 is arbitrary)

    converged = [False] * len(nan_xyz)
    while not all(converged):
        # Step 2: Placeholders are set back to missing for one variable/column
        dependent_col = int(np.random.choice(list(cols_missing)))
        missing_xs = [int(x) for x, y, value in nan_xyz if y == dependent_col]

        # Step 3: Perform linear regression using the other variables
        x_train, y_train = [], []
        for x_i in (x_i for x_i in range(len(data)) if x_i not in missing_xs):
            x_train.append(np.delete(data[x_i], dependent_col))
            y_train.append(data[x_i][dependent_col])
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Step 4: Missing values for the missing variable/column are replaced
        # with predictions from our new linear regression model
        # For null indices with the dependent column that was randomly chosen
        for i, z in enumerate(nan_xyz):
            x_i = z[0]
            y_i = z[1]
            value = data[x_i, y_i]
            if y_i == dependent_col:
                # Row 'x' without the nan value
                new_value = model.predict([np.delete(data[x_i], dependent_col)])
                data[x_i][y_i] = new_value.reshape(1, -1)
                if value == 0.0:
                    delta = (new_value-value)/0.01
                else:
                    delta = (new_value-value)/value
                converged[i] = abs(delta) < eps
    return data

