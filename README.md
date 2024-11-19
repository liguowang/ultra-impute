## Dependencies
The following packages will be automatically installed

1. [numpy](https://numpy.org/)
2. [scipy](https://scipy.org/)
3. [pandas](https://pandas.pydata.org/)
4. [fancyimpute](https://github.com/iskandr)
5. [scikit-learn](https://scikit-learn.org/stable/)
6. [tensorflow](https://www.tensorflow.org/)
7. [keras](https://keras.io/)

## Installation

### 1. Create Virtual Environments (Note: `venv` is available in Python 3.3 and later. You can also use [virtualenv](https://packaging.python.org/en/latest/key_projects/#virtualenv)). This step is optional.

 `$ python3 -m venv my_env` # create a virtual environment called `my_env`

 `$ source my_env/bin/activate` # activates `my_env`

### 2. upgrade [pip](https://pip.pypa.io/en/stable/)

`pip install --upgrade pip`

### 3. Install ultra-impute

`pip install git+https://github.com/liguowang/ultra-impute.git`

or

`pip install --upgrade git+https://github.com/liguowang/ultra-impute.git`


### Note

You will have to run `$ source my_env/bin/activate` to activate `my_env` every time you login
to your computer, unless you add this command to your ~/.bash_profile or ~/.bashrc file.

## Ultra-Impute contains these functions
| Function      | Description |
| --------------| ------- |
| `fill_trend()`|  Replace missing values using methods such as `mean`, `median`, minimum (`min`), maximum (`max`), backfill (`bfill`), or forward fill (`ffill`). |
| `fill_rand()` | Fill missing values with values randomly selected from the same row or column. |
| `fill_mw()`   | Replace missing values with mean values computed using a moving window along rows or columns. |
| `fill_fKNN()` | Replace missing values with values computed from the weighted mean of K-nearest neighbors (adapted from the `impyute` library). |
| `fill_KNN()` | Replace missing values with values computed as the mean of K-nearest neighbors (same as `scikit-learn`). |
| `fill_ref()` | Replace missing values with values computed as the mean of K-nearest neighbors. Unlike fKNN and KNN, the nearest neighbors are identified from an external reference dataset. |
| `fill_EM()` | Impute missing values using the Expectation-Maximization (EM) algorithm. |
| `fill_Buck()` | Impute missing values using [Buck's method](https://www.jstor.org/stable/2984099), a statistical technique for handling incomplete data.|
| `fill_NNM()`| Impute missing values using NuclearNormMinimization. Only work for small dataset. |
| `fill_SoftImpute()`| Same as scikit-learn's `SoftImpute` method. |
| `fill_IterativeSVD()`| Same as scikit-learn's `IterativeSVD` method. |
| `fill_IterativeImputer()`| Same as scikit-learn's `IterativeImputer` method. |
| `fill_MatrixFactorization()`| Same as fancyimpute's `MatrixFactorization` method. |
| `fill_RF()`| Imputes missing data using Random Forest. |
| `fill_more()`| Imputes missing values using DNN or multi-output regressors (KNN, RF, SVR). |
