#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:22:56 2024
@author: Liguo Wang (WangLiguo78@Gmail.com)
"""
import sys,os
import numpy as np
import pandas as pd
from impyutelib import nan_indices, apply_method, random_impute, moving_window
from impyutelib import fKNN, em, buck_iterative, external_ref
from fancyimpute import NuclearNormMinimization, SoftImpute, IterativeSVD
from fancyimpute import IterativeImputer, MatrixFactorization
from misspylib import MissForest
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import Input
from keras.metrics import RootMeanSquaredError

from util import cluster_cols
from scipy.stats import chi2_contingency

__all__ = ["MissFiller"]


class MissFiller:

    def __init__(self, data):
        """
        MissFiller can be initialized with a pandas DataFrame, numpy ndarray,
        or dictionary. Dictionaries and ndarrays will automatically be
        converted to a DataFrame.
        
        Examples
        --------
        >>> d1 = {
            'A':[0, 5, 10, 15, 20],
            'B': [1, 6, np.nan, 16, 21],
            'C':[np.nan, 7, 12, 17, 22],
            'D':[3, 8, 13, 18, 23]
            }
        >>> d2 = np.array([
            [ 0.,  1., np.nan,  3.],
            [ 5.,  6.,  7.,  8.],
            [10., np.nan, 12., 13.],
            [15., 16., 17., 18.],
            [20., 21., 22., 23.]
            ])
        >>> d3= pd.DataFrame(d1)
        >>> d4 = pd.DataFrame(d2, columns=['A','B ','C','D'])
        >>>
        >>> mf1 = MissFiller(d1)
        >>> mf2 = MissFiller(d2)
        >>> mf3 = MissFiller(d3)
        >>> mf4 = MissFiller(d4)
        >>>
        >>> # mf1, mf2, mf3 and mf4 all contain the same data.
        >>> mf1.df
            A     B     C   D
        0   0   1.0   NaN   3
        1   5   6.0   7.0   8
        2  10   NaN  12.0  13
        3  15  16.0  17.0  18
        4  20  21.0  22.0  23
        """
        
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            self.df = pd.DataFrame(data)
        elif isinstance(data, dict) and isinstance(next(iter(data.values())), list):
            self.df = pd.DataFrame(data)
        else:
            raise Exception("Not a valid input. Acceptes pd.DataFrame, np.ndarray or dict.")
        self.nrow = self.df.shape[0]
        self.ncol = self.df.shape[1]
        self.size = self.df.size
        self.dimension = self.df.shape
        self.row_ids = self.df.index
        self.col_ids = self.df.columns
        self.na_count = int(self.df.isna().sum().sum())


    def __str__(self):
        return "DataFrame with %d rows and %d columns." % self.df.shape


    def get_dataframe(self):
        """
        Returns
        -------
        pd.DataFrame
        """
        return self.df


    def get_na_indices(self):
        """
        Returns the coordinates (x, y) of the missing values.
        
        Examples
        --------
        >>> mf1.get_na_indices()
        array([[0, 2],
              [2, 1]])
        
        Returns
        -------
        Array of lists.
        """
        return nan_indices(self.df.to_numpy())


    def count_na(self):
        """
        Calculates the total number of missing values in the DataFrame.
        
        Returns
        -------
        int
            Total number of missing values.
        """
        return self.df.isna().sum().sum()


    def count_row_na(self):
        """
        Counts the missing values in each row.

        Examples
        --------
        >>> mf1.count_row_na()
        0    1
        1    0
        2    1
        3    0
        4    0
        dtype: int64

        Returns
        -------
        pd.Series
            Missing values per row.
        """
        return self.df.isna().sum(axis=1)


    def count_col_na(self):
        """
        Counts the missing values in each column.
        
        Examples
        --------
        >>> mf1.count_col_na()
        A    0
        B    1
        C    1
        D    0
        
        Returns
        -------
        pd.Series
            Missing values per column.
        """
        return self.df.isna().sum(axis=0)


    def remove_na(self, n_non_miss='all', axis=0):
        """
        Remove missing values.
        
        Parameters
        ----------
        axis : {0 , 1}
            0: remove rows with missing values.
            1: remove columns with missing values. 
            Default is 0.
        
        n_non_miss : {'all', float}
            Specifies the required number of non-missing values.
            If 0 < n_non_miss < 1, it is interpreted as a fraction. For
            example, if n_non_miss = 0.85, rows or columns with up to 15%
            missing values will be removed.
        
        Examples
        --------
        >>> mf1.remove_na()
            A     B     C   D
        1   5   6.0   7.0   8
        3  15  16.0  17.0  18
        4  20  21.0  22.0  23
        >>> mf1.remove_na(axis=1)
            A   D
        0   0   3
        1   5   8
        2  10  13
        3  15  18
        4  20  23
        
        Returns
        -------
        pd.DataFrame
        """
        if n_non_miss == 'all':
            if axis == 0:
                n_non_miss = self.ncol
            elif axis == 1:
                n_non_miss = self.nrow
        elif n_non_miss > 0 and n_non_miss < 1:
            if axis == 0:
                n_non_miss = n_non_miss * self.ncol
            elif axis == 1:
                n_non_miss = n_non_miss * self.nrow
        else:
            raise ValueError("Invalid value.")
        
        return self.df.dropna(axis=axis, thresh = int(n_non_miss))


    def insert_na(self, n_miss, seed=123):
        """
        Insert missing values into the existing DataFrame. If n_miss = 100 
        and the DataFrame already has more than 100 missing values (NAs), 
        the function will return the original DataFrame unchanged. If 
        n_miss = 100 and the DataFrame has 20 NAs, the function will insert 
        80 additional NAs into random locations.
        
        Parameters
        ----------
        n_miss : int
            Number of missign values inserted into the dataframe.
        
        seed : int
            Seed to initiate a random number generator.
        
        Returns
        -------
        pd.DataFrame
        
        Examples
        --------
        >>> mf1.insert_na(n_miss = 5)
              A     B     C     D
        0   0.0   1.0   NaN   3.0
        1   5.0   6.0   7.0   NaN
        2   NaN   NaN   NaN  13.0
        3  15.0  16.0  17.0  18.0
        4  20.0  21.0  22.0  23.0
        """
        np.random.seed(seed)
        input_df = self.df.copy()
        nrow,ncol = self.nrow, self.ncol
        previous_na_count = self.na_count
        if n_miss <= 0:
            return input_df
        elif n_miss <= previous_na_count:
            return input_df
        elif n_miss >= nrow*ncol:
            out_df = input_df.replace(input_df.values, np.nan)
        else:
            na_count = 0
            n_miss_needed = int(n_miss - previous_na_count)
            tmp = input_df.to_numpy()
            while(1):
                if na_count >= n_miss_needed:
                    break
                x = np.random.choice(nrow)
                y = np.random.choice(ncol)
                if not np.isnan(tmp[x][y]):
                    tmp[x][y] = np.nan
                    na_count += 1
            out_df = pd.DataFrame(
                tmp, index=input_df.index, columns=input_df.columns)
        return out_df


    def replace_na(self, value, axis=None):
        """
        Replaces missing values with a specified value (can be an integer,
        float, or string).
        
        Examples
        --------
        >>> mf1.replace_na(100)
            A      B      C   D
        0   0    1.0  100.0   3
        1   5    6.0    7.0   8
        2  10  100.0   12.0  13
        3  15   16.0   17.0  18
        4  20   21.0   22.0  23
        
        Returns
        -------
        pd.DataFrame
        """
        return self.df.fillna(value, axis=axis)


    def fill_trend(self, axis=0, method='mean'):
        """
        Replaces missing values with one of the following: 'min', 'max',
        'mean', 'median', 'bfill' (backfill), or 'ffill' (forward fill).

        Parameters
        ----------
        axis : {0, 1}
            0: calculate mean/median/min/max or find the forward-fill/back-fill
               value along the columns 
            1: calculate mean/median/min/max or find the forward-fill/back-fill
               value along the rows.
            The default is 0.
        
        method : {'mean', 'median', 'min', 'max', 'bfill', 'ffill'}
            mean, median, min, max: 
                refers to pandas's documentations.
            bfill (back fill): Fills a missing value with the next value.
                If axis=0, "next" refers to the value directly below the 
                missing value. As a result, missing values at the bottom of 
                the column will not be filled.
                If axis=1, "next" refers to the value to the right of the 
                missing value. Consequently, missing values on the rightmost 
                side of the row will not be filled.
            ffill (forward fill): Fills a missing value with the previous value.
                If axis=0, "previous" refers to the value directly above the 
                missing value. Therefore, missing values at the top of the 
                column will not be filled.
                If axis=1, "previous" refers to the value to the left of the 
                missing value. Thus, missing values on the leftmost side of 
                the row will not be filled.
        
        Examples
        --------
        >>> mf1.fill_trend(method='mean')
                A     B     C   D
            0   0   1.0  14.5   3
            1   5   6.0   7.0   8
            2  10  11.0  12.0  13
            3  15  16.0  17.0  18
            4  20  21.0  22.0  23
        >>> mf1.fill_trend(method='mean', axis=1)
                  A          B          C     D
            0   0.0   1.000000   1.333333   3.0
            1   5.0   6.000000   7.000000   8.0
            2  10.0  11.666667  12.000000  13.0
            3  15.0  16.000000  17.000000  18.0
            4  20.0  21.000000  22.000000  23.0
        >>> mf1.fill_trend(method='bfill')
                A     B     C   D
            0   0   1.0   7.0   3
            1   5   6.0   7.0   8
            2  10  16.0  12.0  13
            3  15  16.0  17.0  18
            4  20  21.0  22.0  23
        
        Returns
        -------
        pd.DataFrame
        """
        if axis == 0:
            input_df = self.df
            tmp = apply_method(input_df, method)
            out_df = input_df.fillna(tmp)
        elif axis == 1:
            input_df = self.df.T
            tmp = apply_method(input_df, method)
            out_df = input_df.fillna(tmp).T
        return out_df


    def fill_rand(self, axis=0):
        """
        Replaces missing values with random values selected from the
        corresponding row or column.
        
        Parameters
        ----------
        axis : {0, 1}
            0: chosen a random value from the column.
            1: chosen a random value from the row.
            Default is 0.
        
        Examples
        --------
        >>> mf1.fill_rand()
              A     B     C     D
        0   0.0   1.0  17.0   3.0
        1   5.0   6.0   7.0   8.0
        2  10.0  21.0  12.0  13.0
        3  15.0  16.0  17.0  18.0
        4  20.0  21.0  22.0  23.0
        
        Returns
        -------
        pd.DataFrame
        """
        if axis == 0:
            input_df = self.df
            return random_impute(input_df)
        elif axis == 1:
            input_df = self.df.T
            return random_impute(input_df).T


    def fill_mw(self, axis=0, nindex=None, wsize=5, errors="coerce", 
                func=np.mean):
        """
        Replaces missing values with values calculated from moving windows
        along rows or columns.
        
        Parameters
        ----------
        axis : {0, 1}
            0: Apply moving windows along the columns.
            1: Apply moving windows along the rows.
            Default is 0.
        
        nindex: int
            Null index. Index of the missing value inside the moving average
            window. This is useful if you wanted to make the imputed value
            skewed toward the left or right side. 
            0:  only take the average of values from the right side of the
                missing value.
            -1: only take the average of values from the left side of the
                missing value.
        
        wsize: int
            Size of the moving average window/area of values being used
            for each local imputation. This number includes the missing value.
        
        errors: {"raise", "coerce", "ignore"}
            Errors will occur with the indexing of the windows - for example 
            if there is a nan at data[x][0] and `nindex` is set to -1 or there
            is a nan at data[x][-1] and `nindex` is set to 0. `"raise"` will
            raise an error, `"coerce"` will try again using an nindex set to
            the middle and `"ignore"` will just leave it as a nan.
        
        Examples
        --------
        >>> mf1.fill_mw()
              A          B     C     D
        0   0.0   1.000000   9.5   3.0
        1   5.0   6.000000   7.0   8.0
        2  10.0   7.666667  12.0  13.0
        3  15.0  16.000000  17.0  18.0
        4  20.0  21.000000  22.0  23.0
        >>> mf1.fill_mw(axis=1)
              A          B          C     D
        0   0.0   1.000000   1.333333   3.0
        1   5.0   6.000000   7.000000   8.0
        2  10.0  11.666667  12.000000  13.0
        3  15.0  16.000000  17.000000  18.0
        4  20.0  21.000000  22.000000  23.0
        
        Returns
        -------
        pd.DataFrame
        """
        if axis == 1:
            input_df = self.df
            out_df = moving_window(input_df, nindex = nindex, wsize=wsize, 
            errors=errors, func=func)
        elif axis == 0:
            input_df = self.df.T
            out_df = moving_window(input_df, nindex = nindex, wsize=wsize, 
            errors=errors, func=func).T
        return out_df


    def fill_fKNN(self, method='mean', axis=1, k=3, eps=0, p=2, 
                  distance_upper_bound=np.inf, leafsize=10):
        """
        Impute missing values using the K-nearest neighbors (kNN) algorithm.
        First, apply an initial imputation function (e.g., mean imputation)
        to handle missing data, creating a complete array. Use this array
        to construct a KDTree to identify the nearest neighbors. After finding
        the k nearest neighbors, compute a weighted average of these neighbors
        based on their distances to perform the final imputation.
        
        Parameters
        ----------
        axis : int, optional
            Must be 0 or 1. 
            0: Use *column* mean/median/min/max/bfill/ffill for initial 
               imputation. Search *rows* for k nearest neighbours.
            1: Use *row* mean/median/min/max/bfill/ffill for initial 
               imputation. Search *columns* for k nearest neighbours.
            The default is 1 (see Notes below).

        method : {'mean', 'median', 'min', 'max', 'bfill', 'ffill'}
            Initial imputation method. See fill_trend() for details.
            The default is 'mean'

        k: int, optional
            Parameter used for method querying the KDTree class object. Number
            of neighbours used in the KNN query.

        eps: nonnegative float, optional
            Parameter used for method querying the KDTree class object. From
            the SciPy docs: "Return approximate nearest neighbors; the kth
            returned value is guaranteed to be no further than (1+eps) times
            the distance to the real kth nearest neighbor".

        p : float, 1<=p<=infinity, optional
            Parameter used for method querying the KDTree class object.
            Straight from the SciPy docs: "Which Minkowski p-norm to use. 1 is
            the sum-of-absolute-values Manhattan distance 2 is the usual
            Euclidean distance infinity is the maximum-coordinate-difference
            distance".

        distance_upper_bound : nonnegative float, optional
            Parameter used for method querying the KDTree class object.
            Straight from the SciPy docs: "Return only neighbors within this 
            instance. This is used to prune tree searches, so if you are doing
            a series of nearest-neighbor queries, it may help to supply the
            distance to the nearest neighbor of the most recent point.

        leafsize: int, optional
            Parameter used for construction of the `KDTree` class object.
            Straight from the SciPy docs: "The number of points at which the
            algorithm switches over to brute-force. Has to be positive".

        Notes
        -----
        In biomedical research, it is common practice to organize features 
        (e.g., genes, CpGs, genomic regions) as rows and samples (e.g., 
        patients, cells) as columns. An example is shown below:
            cg_ID   TCGA-BC-A10Q    TCGA-BC-A10R    TCGA-BC-A10S    TCGA-BC-A10T    TCGA-BC-A10U
            cg00000029      0.3469  0.387   0.3428  0.3064  0.3939
            cg00000165      NA      0.1656  0.1212  0.1171  0.1626
            cg00000236      0.8479  NA      0.8647  0.8918  0.8674
            cg00000289      0.6658  0.5231  0.6022  0.7026  0.7297
            cg00000292      0.6913  0.752   0.6212  0.751   0.7616
            cg00000363      0.7589  0.6407  0.6119  0.6595  0.6155
            cg00000622      0.015   0.0137  0.0277  0.0139  0.0141
            cg00000658      0.8987  0.8959  0.6375  0.881   0.8795
            cg00000714      0.1591  0.1372  0.2413  0.173   0.2007
            cg00000721      0.9491  0.9464  0.8963  0.9413  0.9512
            cg00000734      0.0492  0.0506  0.0477  0.0695  0.0643
            ...
            In this case, we recommend to set axis = 1.

        Examples
        --------
        >>> mf1.fill_fKNN()
              A          B          C     D
        0   0.0   1.000000   1.473573   3.0
        1   5.0   6.000000   7.000000   8.0
        2  10.0  11.535419  12.000000  13.0
        3  15.0  16.000000  17.000000  18.0
        4  20.0  21.000000  22.000000  23.0

        Returns
        ----------
        pd.DataFrame
        """

        if axis == 1:
            na_ind = nan_indices(self.df.T.to_numpy())
        elif axis == 0:
            na_ind = nan_indices(self.df.to_numpy())
        else:
            raise ValueError("axis must be 0 or 1.")
        
        #pre-impute
        input_df = self.fill_trend(method=method, axis=axis)

        if axis == 1:
            input_df = input_df.T
            out_df = fKNN(input_df, na_locations = na_ind, k=k, eps=eps, p=p,
                 distance_upper_bound=distance_upper_bound, 
                 leafsize=leafsize)
            out_df = out_df.T
        elif axis == 0:
            out_df = fKNN(input_df, na_locations = na_ind, k=k, eps=eps, p=p,
                 distance_upper_bound=distance_upper_bound, 
                 leafsize=leafsize)
        else:
            raise ValueError("axis only accepts 0 or 1.")
        return out_df


    def fill_ref(self, ref, axis=1, k=3, eps=0, p=2, 
                  distance_upper_bound=np.inf, leafsize=10):
        """
        The algorithm is similar to fKNN imputation, where the k-nearest 
        neighbors are searched from the external reference.
        
        Parameters
        ----------
        axis : int, optional
            Must be 0 or 1. 
            0: Use *column* mean/median/min/max/bfill/ffill for initial 
               imputation. Search *rows* for k nearest neighbours.
            1: Use *row* mean/median/min/max/bfill/ffill for initial 
               imputation. Search *columns* for k nearest neighbours.
            The default is 1 (see Notes below).
        
        ref: pd.DataFrame
            The reference DataFrame must NOT contain any missing values.
            When axis=1: The reference DataFrame should be a superset of the
            dataset being imputed with respect to row names (index). 
            Otherwise, rows that cannot be found in the reference may be 
            removed from the results.
            When axis=0: The reference DataFrame should be a superset of the
            dataset being imputed with respect to column names. Otherwise,
            columns that cannot be found in the reference may be removed from 
            the results.

        k: int, optional
            Parameter used for method querying the KDTree class object. Number
            of neighbours used in the KNN query.

        eps: nonnegative float, optional
            Parameter used for method querying the KDTree class object. From
            the SciPy docs: "Return approximate nearest neighbors; the kth
            returned value is guaranteed to be no further than (1+eps) times
            the distance to the real kth nearest neighbor".

        p : float, 1<=p<=infinity, optional
            Parameter used for method querying the KDTree class object.
            Straight from the SciPy docs: "Which Minkowski p-norm to use. 1 is
            the sum-of-absolute-values Manhattan distance 2 is the usual
            Euclidean distance infinity is the maximum-coordinate-difference
            distance".

        distance_upper_bound : nonnegative float, optional
            Parameter used for method querying the KDTree class object.
            Straight from the SciPy docs: "Return only neighbors within this 
            instance. This is used to prune tree searches, so if you are doing
            a series of nearest-neighbor queries, it may help to supply the
            distance to the nearest neighbor of the most recent point.

        leafsize: int, optional
            Parameter used for construction of the `KDTree` class object.
            Straight from the SciPy docs: "The number of points at which the
            algorithm switches over to brute-force. Has to be positive".

        Returns
        -------
        pd.DataFrame.

        """
        # Search for columns from the external reference for "K nearest neighbours"
        if axis == 1:
            names_data = self.df.index
            names_ref = ref.index
            names_common = list(set(names_data) & set(names_ref))
            ref_df = ref.loc[names_common] 
            input_df = self.df.loc[names_common] 
            row_names = input_df.index
            col_names = input_df.columns
            input_df = input_df.T.to_numpy() #transpose input
            na_ind = nan_indices(input_df)
            ref_df = ref_df.T.to_numpy() #transpose reference
            out = external_ref(input_df,
                                  na_locations = na_ind, 
                                  ref_data = ref_df,
                                  k=k, eps=eps, p=p,
                                  distance_upper_bound=distance_upper_bound, 
                                  leafsize=leafsize)
            out_df = pd.DataFrame(np.transpose(out), index=row_names, 
                                  columns = col_names)
        # Search for rows from the external reference for "K nearest neighbours"
        elif axis == 0:
            names_data = self.df.columns
            names_ref = ref.columns
            names_common = list(set(names_data) & set(names_ref))
            ref_df = ref[names_common] 
            input_df = self.df[names_common] 
            row_names = input_df.index
            col_names = input_df.columns
            input_df = input_df.to_numpy()
            na_ind = nan_indices(input_df)
            ref_df = ref_df.to_numpy()
            out = external_ref(input_df,
                                  na_locations = na_ind, 
                                  ref_data = ref_df,
                                  k=k, eps=eps, p=p,
                                  distance_upper_bound=distance_upper_bound, 
                                  leafsize=leafsize)
            out_df = pd.DataFrame(out, index=row_names, 
                                  columns = col_names)

        else:
            raise ValueError("axis only accepts 0 or 1.")
        return out_df


    def fill_KNN(self, axis=1, method='mean', **kwargs):
        """
        Use sklearn's KNNImputer function with some improvements.
        
        Parameters
        ----------
        axis : {0, 1}
            0: Use *column* mean for initial imputation. Search *rows* for k
                nearest neighbours.
            1: Use *row* mean for initial imputation. Search *columns* for k
                nearest neighbours.
            The default is 1

        method : {'mean', 'median', 'min', 'max', 'bfill', 'ffill'}
            Initial imputation method. See fill_trend() for details.
            The default is 'mean'.

        Examples
        --------
        >>> mf1.fill_KNN()
              A          B          C     D
        0   0.0   1.000000   1.333333   3.0
        1   5.0   6.000000   7.000000   8.0
        2  10.0  11.666667  12.000000  13.0
        3  15.0  16.000000  17.000000  18.0
        4  20.0  21.000000  22.000000  23.0

        Returns
        -------
        pd.DataFrame.
        """
        #input_df = self.fill_trend(method=method, axis=axis)
        input_df = self.df

        imputer = KNNImputer(**kwargs)
        #impute on rows
        if axis == 1:
            input_df = input_df.T
            after = imputer.fit_transform(input_df)
            out_df = pd.DataFrame(after, index = input_df.index,
                                  columns = input_df.columns).T
            #output_df = output_df.round(args.decimal)
        elif axis == 0:
            after = imputer.fit_transform(input_df)
            out_df = pd.DataFrame(after, index = input_df.index,
                                  columns = input_df.columns)
        else:
            raise ValueError("axis only accepts 0 or 1.")
        return out_df
            


    def fill_EM(self, axis=1, eps=0.001):
        """
        Imputes missing data using the Expectation-Maximization (EM) algorithm.
        * E-step: Calculates the expected log-likelihood of the complete data.
        * M-step: Optimizes the parameters to maximize the log-likelihood of 
          the complete data.

        Parameters
        ----------
        axis : {0, 1}
            0: perform EM on non-missing values from the same column.
            1: perform EM on non-missing values from the same row.
            Default is 1.

        eps : float, optional
            The amount of minimum change between iterations to break, if 
            relative change < eps, converge. 
            relative change = abs(current - previous) / previous

        Examples
        --------
        >>> mf1.fill_EM()
              A          B          C     D
        0   0.0   1.000000   3.591067   3.0
        1   5.0   6.000000   7.000000   8.0
        2  10.0  11.776261  12.000000  13.0
        3  15.0  16.000000  17.000000  18.0
        4  20.0  21.000000  22.000000  23.0

        Returns
        -------
        pd.DataFrame

        """
        if axis == 1:
            out_df = em(self.df.T, eps=eps).T
        elif axis == 0:
            out_df = em(self.df, eps=eps)
        else:
            raise Exception("axis must be 0 or 1.")
        return out_df


    def fill_Buck(self, axis=0, eps=0.001):
        """
        Iterative Variant of Buck's Method
        
        The variable to be regressed is selected randomly at each iteration.
        The iterative EM-like process continues until the change in 
        predictions, compared to the previous iteration, is less than 10% 
        for all columns with missing values.
        
        Reference
        ---------
        S.F. Buck, "A Method of Estimation of Missing Values in Multivariate 
        Data Suitable for Use with an Electronic Computer," Journal of the 
        Royal Statistical Society: Series B (Methodological), Vol. 22, No. 2 
        (1960), pp. 302-306.

        Parameters
        ----------
        axis : {0, 1}
            0: perform Buck's method on columns.
            1: erform Buck's method on rows.
            The default is 0.

        eps : float, optional
            The amount of minimum change between iterations to break, if 
            relative change < eps, converge. 
            relative change = abs(current - previous) / previous

        Examples
        --------
        >>> mf1.fill_Buck()
              A     B     C     D
        0   0.0   1.0   2.0   3.0
        1   5.0   6.0   7.0   8.0
        2  10.0  11.0  12.0  13.0
        3  15.0  16.0  17.0  18.0
        4  20.0  21.0  22.0  23.0

        Returns
        -------
        pd.DataFrame

        """
        if axis == 1:
            out_df = buck_iterative(self.df.T, eps = eps).T
        elif axis == 0:
            out_df = buck_iterative(self.df, eps = eps)
        else:
            raise Exception("axis must be 0 or 1.")
        return out_df


    def fill_NNM(self, require_symmetric_solution=False, min_value=None, 
                 max_value=None, error_tolerance=0.001, max_iters=5000):
        """
        Impute missing values using NuclearNormMinimization.

        Examples
        --------
        >>> mf1.fill_NNM()
              A     B          C     D
        0   0.0   1.0   1.998666   3.0
        1   5.0   6.0   7.000000   8.0
        2  10.0  11.0  12.000000  13.0
        3  15.0  16.0  17.000000  18.0
        4  20.0  21.0  22.000000  23.0

        Returns
        -------
        pd.DataFrame
        
        NOTE:
        ----
        This process is very slow, especially for larger dataset.
        You will get overflow error for large matrix.
        """
        X_filled = NuclearNormMinimization(require_symmetric_solution = require_symmetric_solution, 
                                           min_value = min_value,
                                           max_value = max_value,
                                           error_tolerance = error_tolerance,
                                           max_iters = max_iters,
                                           ).fit_transform(self.df.to_numpy())
        out_df = pd.DataFrame(
            X_filled, index=self.df.index, columns=self.df.columns)
        return out_df


    def fill_SoftImpute(self, shrinkage_value=None, 
                        convergence_threshold=0.001, 
                        max_iters=500, max_rank=None, n_power_iterations=1, 
                        init_fill_method='zero', min_value=None, 
                        max_value=None,normalizer=None):
        """
        Matrix completion by iterative soft thresholding of SVD decompositions.
        Similar to R softImpute package.

        Examples
        --------
        >>> mf1.fill_SoftImpute()
              A          B          C     D
        0   0.0   1.000000   1.608209   3.0
        1   5.0   6.000000   7.000000   8.0
        2  10.0  10.776869  12.000000  13.0
        3  15.0  16.000000  17.000000  18.0
        4  20.0  21.000000  22.000000  23.0

        Returns
        -------
        pd.DataFrame
        
        """
        X_filled = SoftImpute(shrinkage_value = shrinkage_value, 
                              convergence_threshold = convergence_threshold,
                              max_iters = max_iters, 
                              max_rank = max_rank,
                              n_power_iterations = n_power_iterations,
                              init_fill_method = init_fill_method,
                              min_value = min_value,
                              max_value = max_value,
                              normalizer = normalizer
                              ).fit_transform(self.df.to_numpy())
        out_df = pd.DataFrame(X_filled, index=self.df.index, columns=self.df.columns)
        return out_df


    def fill_IterativeSVD(self, rank=10, convergence_threshold=0.001, 
                          max_iters=200, gradual_rank_increase=True, 
                          svd_algorithm='arpack', init_fill_method='zero', 
                          min_value=None, max_value=None):
        """
        Matrix completion by iterative low-rank SVD decomposition. The input
        dataframe must have at least 5 columns.

        Examples
        --------
        >>> d5 = {
        ... 'A':[0, 5, 10, 15, 20],
        ... 'B': [1, 6, np.nan, 16, 21],
        ... 'C':[np.nan, 7, 12, 17, 22],
        ... 'D':[3, 8, 13, 18, 23],
        ... 'E':[4, 9, 14, 19, 24]
        ... }
        >>> mf5 = MissFiller(d5)
        >>> mf5.fill_IterativeSVD()
              A          B          C     D     E
        0   0.0   1.000000   2.039475   3.0   4.0
        1   5.0   6.000000   7.000000   8.0   9.0
        2  10.0   9.519816  12.000000  13.0  14.0
        3  15.0  16.000000  17.000000  18.0  19.0
        4  20.0  21.000000  22.000000  23.0  24.0

        Returns
        -------
        pd.DataFrame
        
        """
        X_filled = IterativeSVD(rank = rank, 
                              convergence_threshold = convergence_threshold,
                              max_iters = max_iters, 
                              gradual_rank_increase = gradual_rank_increase,
                              svd_algorithm = svd_algorithm,
                              init_fill_method = init_fill_method,
                              min_value = min_value,
                              max_value = max_value
                              ).fit_transform(self.df.to_numpy())
        out_df = pd.DataFrame(X_filled, index=self.df.index, columns=self.df.columns)
        return out_df


    def fill_IterativeImputer(self):
        """
        A strategy for imputing missing values by modeling each feature with 
        missing values as a function of other features in a round-robin fashion.
        Same as MICE (Multiple Imputation by  chained equations) in R.


        Examples
        --------
        >>> mf1.fill_IterativeImputer()
              A     B     C     D
        0   0.0   1.0   2.0   3.0
        1   5.0   6.0   7.0   8.0
        2  10.0  11.0  12.0  13.0
        3  15.0  16.0  17.0  18.0
        4  20.0  21.0  22.0  23.0

        Returns
        -------
        pd.DataFrame

        """
        X_filled = IterativeImputer().fit_transform(self.df.to_numpy())
        out_df = pd.DataFrame(X_filled, index=self.df.index, columns=self.df.columns)
        return out_df


    def fill_MatrixFactorization(self, rank=40, learning_rate=0.01, 
                                 max_iters=500, shrinkage_value=0, 
                                 min_value=None, max_value=None):
        """
        Direct factorization of the incomplete matrix into low-rank U and V, 
        with per-row and per-column biases, as well as a global bias.

        Examples
        --------
        >>> mf1.fill_MatrixFactorization()
              A          B         C     D
        0   0.0   1.000000   2.32574   3.0
        1   5.0   6.000000   7.00000   8.0
        2  10.0  11.003153  12.00000  13.0
        3  15.0  16.000000  17.00000  18.0
        4  20.0  21.000000  22.00000  23.0

        Returns
        -------
        pd.DataFrame
        
        """
        X_filled = MatrixFactorization(rank = rank,
                                       learning_rate = learning_rate,
                                       max_iters = max_iters,
                                       shrinkage_value = shrinkage_value,
                                       min_value = min_value,
                                       max_value = max_value
                                       ).fit_transform(self.df.to_numpy())
        out_df = pd.DataFrame(X_filled, index=self.df.index, columns=self.df.columns)
        return out_df


    def fill_RF(self, max_iter=500, decreasing=False, missing_values=np.nan,
                 copy=True, n_estimators=100, criterion=('squared_error', 'gini'),
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=1.0,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
                 verbose=0, warm_start=False, class_weight=None):
        """
        Missing value imputation using Random Forests.

        Examples
        --------
        >>> mf1.fill_RF()
              A     B     C     D
        0   0.0   1.0   8.6   3.0
        1   5.0   6.0   7.0   8.0
        2  10.0   6.4  12.0  13.0
        3  15.0  16.0  17.0  18.0
        4  20.0  21.0  22.0  23.0

        Returns
        -------
        pd.DataFrame
        
        """
        imputer = MissForest(max_iter = max_iter,
                             decreasing = decreasing,
                             missing_values = missing_values,
                             copy = copy,
                             n_estimators = n_estimators,
                             criterion = criterion,
                             max_depth = max_depth,
                             min_samples_split = min_samples_split,
                             min_samples_leaf = min_samples_leaf,
                             min_weight_fraction_leaf = min_weight_fraction_leaf,
                             max_features = max_features,
                             max_leaf_nodes = max_leaf_nodes,
                             min_impurity_decrease = min_impurity_decrease,
                             bootstrap = bootstrap,
                             oob_score = oob_score,
                             n_jobs = -1,
                             random_state = random_state,
                             verbose = verbose,
                             warm_start = warm_start,
                             class_weight = class_weight)
        X_filled = imputer.fit_transform(self.df.to_numpy())
        out_df = pd.DataFrame(X_filled, index=self.df.index, columns=self.df.columns)
        return out_df

    def fill_morel(self, group=None, decimal=5, initial_model = 'Buck',
                  second_model = 'RF', niter = 10, seed=100, n_proc=8,
                  train_size=0.75, epochs = 100):
        """
        

        Parameters
        ----------
        group : dict
            Describe the group information of samples. For example:
                {'A':["sample1", "sample2", "sample3"],
                 'B':["sample4", "sample5", "sample6"]}
            Group ID can be arbitrary, but sample IDs must match the column names.
            if set to None, using K-means to detect groups.
        decimal : int, optional
           "Number of decimal places to round. The default is 5.
        initial_model : {'Buck', 'KNN', 'MICE', 'RF'}
            Model used for initial imputation of the random missings.
                Buck : Buck's method
                KNN : K-Nearest Neighbors
                MICE : Multiple Imputation by Chained Equations
                RF : Random Forest
        second_model : {'RF', 'KNN', 'SVR', 'DNN'}
            Model used for imputing the blocky/systematic missings.
                RF : Random Forest
                KNN : K-Nearest Neighbors
                SVR : Support Vector Regression
                DNN : Deep Neural network
        
        Below parameters are for {'RF', 'KNN', 'SVR'}
        niter : int
            Repeating times of the secondary imputation.
        seed : int, optional
            Seed used to initialize a pseudorandom number generator.
            The default is 1234.
        n_proc : int, optional
            Number of processors to use. The default is 8.
        train_size : float, optional
            Fraction of samples used as training dataset.Used to split the
            samples into "training" and "testing". The default is 0.75.
        
        Below parameters are for "DNN"
        epochs : int, optional
            the total number of iterations of all the training data in one
            cycle for training the Deep Neural network model.

        Returns
        -------
        result : df.DataFrame

        """
    
        #toal_na =self.na_count
        if group is None:
            print("Binerize sample IDs into two groups using K-means ...", file=sys.stderr)
            group = cluster_cols(self.df)
        
        #Find rows where all values are missing
        all_missing_ind = self.df.isna().all(axis=1)
        df_all_missing = self.df.loc[all_missing_ind]
       
        # Drop %d rows where all values are missing.
        df1 = self.df.dropna(how='all')
       
        all_samples = []
        group_names = sorted(group.keys())
        for g in group_names:
            print("Group \"%s\" contains %d samples" % (str(g), len(group[g])), file=sys.stderr)
            for s in group[g]:
                print("\t" + s, file=sys.stderr)
            all_samples.extend(group[g])
       
        if  len(all_samples) != len(set(all_samples)):
            print("Sample IDs are not unique", file=sys.stderr)
            sys.exit()

        g0_samples = group[group_names[0]]
        g1_samples = group[group_names[1]]
        
        # exclude samples from data_file if they are not in the group_file
        used_df = df1[all_samples]
        df_all_missing = df_all_missing[all_samples]
        used_df_g0 = used_df[g0_samples]
        used_df_g1 = used_df[g1_samples]
        
        # list of bool values indicating each row is "ALL NA" or not
        g0_all_NA_ind = used_df_g0.isnull().all(axis=1) #g0: Entire row are "NAs"
        g1_all_NA_ind = used_df_g1.isnull().all(axis=1) #g1: Entire row are "NAs"
        print("%d rows in group \"%s\" are complete missing." % (sum(g0_all_NA_ind), group_names[0]), file=sys.stderr)
        print("%d rows in group \"%s\" are complete missing." % (sum(g1_all_NA_ind), group_names[1]), file=sys.stderr)
        
        g0_all_NA = used_df.loc[g0_all_NA_ind].copy() #g0 "block missing" that need imputation
        g1_all_NA = used_df.loc[g1_all_NA_ind].copy() #g1 "block missing" that need imputation
        
        # initial imputation (handle those "random mising")
        # after that, the data will be used to train a model to predict
        # block missing
        used_df_train = used_df.loc[~(g0_all_NA_ind | g1_all_NA_ind)]
        tmp = MissFiller(used_df_train)
        if tmp.na_count > 0:
            print("There are %d sporadic missing values." %  tmp.na_count, file=sys.stderr)
            print("Impute sporadic missing values using Buck's method ...", file=sys.stderr)
            if initial_model == 'Buck':
                print("Initial imputing using Buck's method ...", file=sys.stderr)
                used_df_train = tmp.fill_Buck()
            elif initial_model == 'RF':
                print("Initial imputing using Random Forest ...", file=sys.stderr)
                used_df_train = tmp.fill_RF()
            elif initial_model == 'MICE':
                print("Initial imputing using MICE ...", file=sys.stderr)
                used_df_train = tmp.fill_IterativeImputer()
            else:
                print("Unknown method ...", file=sys.stderr)
        g0_train = used_df_train[g0_samples]
        g1_train = used_df_train[g1_samples]
        
        # secondary imputation (handle those "block mising")
        if  second_model == 'DNN':
                print("Predict missing values in group \"%s\" using deep neural network" % group_names[0], file=sys.stderr)
                input_dim = g1_train.shape[1]
                output_dim = g0_train.shape[1]
                n_neurons = int((input_dim + output_dim)/2) + 1
                print("Split data into training and testing ...", file = sys.stderr)
                X_train, X_test, y_train, y_test = train_test_split(
                g1_train, g0_train, 
                train_size=train_size, random_state=seed
                )
                model = Sequential()
                model.add(Input(shape=(input_dim,)))
                model.add(Dense(n_neurons, kernel_initializer='he_uniform', activation='relu'))
                model.add(Dense(output_dim, kernel_initializer='he_uniform', activation='linear'))
                model.summary()
                model.compile(loss='mae', optimizer='adam', metrics=[RootMeanSquaredError])
                model.fit(X_train, y_train, verbose=0, epochs=epochs)
                results = model.evaluate(X_test, y_test)
                print("Test MAE loss, test RMSE:", results)
                #fit model using all available data
                model.fit(g1_train, g0_train, verbose=0, epochs=epochs)
                pred = model.predict(g0_all_NA[g1_samples])
                g0_all_NA[g0_samples] = pred.round(decimal)

                print("Predict missing values in group \"%s\" using deep neural network" % group_names[1], file=sys.stderr)
                input_dim = g0_train.shape[1]
                output_dim = g1_train.shape[1]
                n_neurons = int((input_dim + output_dim)/2) + 1
                print("Split data into training and testing ...", file = sys.stderr)
                X_train, X_test, y_train, y_test = train_test_split(
                g0_train, g1_train, 
                train_size=train_size, random_state=seed
                )
                model2 = Sequential()
                model2.add(Input(shape=(input_dim,)))
                model2.add(Dense(n_neurons, kernel_initializer='he_uniform', activation='relu'))
                #model2.add(Dense(48, input_shape=(input_dim,), kernel_initializer='he_uniform', activation='relu'))
                model2.add(Dense(output_dim, kernel_initializer='he_uniform', activation='linear'))
                model2.summary()
                model2.compile(loss='mae', optimizer='adam', metrics=[RootMeanSquaredError])
                model.fit(X_train, y_train, verbose=0, epochs=epochs)
                results = model.evaluate(X_test, y_test)
                print("Test MAE loss, test RMSE:", results)
                #fit model using all available data
                model2.fit(g0_train, g1_train, verbose=0, epochs=epochs)
                pred = model2.predict(g1_all_NA[g0_samples])
                g1_all_NA[g1_samples] = pred.round(decimal)
        elif second_model in ['RF', 'KNN', 'SVR']:
            if second_model == 'RF':
                model = MultiOutputRegressor(
                    RandomForestRegressor(max_depth=30, random_state=seed),
                    n_jobs=n_proc)
            elif  second_model == 'KNN':
                model = MultiOutputRegressor(
                    KNeighborsRegressor(n_neighbors=5, n_jobs = n_proc),
                    n_jobs=n_proc)
            elif  second_model == 'SVR':
                model = MultiOutputRegressor(
                    LinearSVR(),
                    n_jobs=n_proc)
                    
            #group "g0" has missing vlaues. Use g1 to predict g0
            print("Predict missing values in group \"%s\" using %s" % (group_names[0], second_model), file=sys.stderr)
            for i in list(range(niter)):
                if len(g0_all_NA) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                    g1_train, g0_train, 
                    train_size=train_size, random_state=seed + i
                    )
                    
                    # Fit on the train data
                    model.fit(X_train, y_train)
                    
                    # Check the prediction score
                    score = model.score(X_test, y_test)
                    print(
                        "Iter %d: the prediction score is %.2f.%%" % (i, round(score*100, 2)), file=sys.stderr)
            
                    # predict
                    if i == 0:
                        pred = model.predict(g0_all_NA[g1_samples])
                    else:
                        pred = (pred + model.predict(g0_all_NA[g1_samples]))/2
                    g0_all_NA[g0_samples] = pred.round(decimal)
                
            #group "g1" has missing vlaues. Use g0 to predict g1
            print("Predict missing values in group \"%s\" using %s" % (group_names[1], second_model), file=sys.stderr)
            for i in list(range(niter)):
                if len(g1_all_NA) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                    g0_train, g1_train, 
                    train_size=train_size, random_state=seed + i
                    )
                    
                    # Fit on the train data
                    model.fit(X_train, y_train)
                    
                    # Check the prediction score
                    score = model.score(X_test, y_test)
                    print(
                        "Iter %d: the prediction score is %.2f.%%" % (i, round(score*100, 2)), file=sys.stderr)
            
                    # predict
                    if i == 0:
                        pred = model.predict(g1_all_NA[g0_samples])
                    else:
                        pred = (pred + model.predict(g1_all_NA[g0_samples]))/2
                    g1_all_NA[g1_samples] = pred.round(decimal)
            
        
        result = pd.concat([used_df_train, g0_all_NA, g1_all_NA, df_all_missing])
        print('Re-order the index as the original dataframe ...', file=sys.stderr)
        result = result.reindex_like(self.df).round(decimal)
        return result


    def missing_test(self, group):
        """
        Test (using the Chi squared tests) if the number of missing values are
        associated with user defined group.

        Parameters
        ----------
        group : dict
            Describe the group information of samples. For example:
                {'A':["sample1", "sample2", "sample3"],
                 'B':["sample4", "sample5", "sample6"]}

        Returns
        -------
        result : df.DataFrame

        """
        table = {}
        results = {}
        group_names = sorted(group.keys())
        for g in group_names:
            print("Group \"%s\" contains %d samples" % (str(g), len(group[g])), file=sys.stderr)
            samples_in_group = group[g]
            for s in samples_in_group:
                print("\t" + s, file=sys.stderr)
            miss = self.df.isna().sum().sum()
            non_miss = self.df.size - miss
            table[g] = [miss, non_miss]

        test_df = pd.DataFrame(table)
        if test_df.size == 4:
            out = chi2_contingency(test_df)
            results['statistic'] = float(out.statistic)
            results['pval'] = float(out.pvalue)
            results['dof'] = float(out.dof)
        return results
