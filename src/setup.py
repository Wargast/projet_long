from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='stereo vision',
    ext_modules=cythonize("census_c.pyx", language_level=3),
    zip_safe=False,
    include_dirs=np.get_include()
)