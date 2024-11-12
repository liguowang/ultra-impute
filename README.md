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
