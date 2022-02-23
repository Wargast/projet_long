from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = [
    Extension(
        "census_c",
        ["census_c.pyx"],
        extra_compile_args=['/openmp']
    )
]


setup(
    name='stereo vision',
    ext_modules=cythonize(ext_modules, language_level=3),
    zip_safe=False,
    include_dirs=np.get_include()
)



