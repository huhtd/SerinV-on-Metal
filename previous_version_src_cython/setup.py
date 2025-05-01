# setup.py
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("cholesky_cython.pyx", language_level=3),
    include_dirs=[np.get_include()]
)


'''to build run this command in same directory
     python setup.py build_ext --inplace'''