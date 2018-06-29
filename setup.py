from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy as np

USE_CYTHON = False
ext = '.pyx' if USE_CYTHON else '.c'

#sys.argv[1:] = ['build_ext', '--inplace']

ext_modules = [
    Extension("pairwisemkl.utilities._sampled_kronecker_products",["pairwisemkl/utilities/_sampled_kronecker_products"+ext])
]

if USE_CYTHON:
    from Cython.Build import cythonize
    ext_modules = cythonize(ext_modules)

setup(
    name = 'pairwisemkl',
    description = 'pairwiseMKL package',
    url = "https://github.com/aalto-ics-kepaco/pairwiseMKL",
    version = "0.1",
    license = "MIT",
    include_dirs = [np.get_include()],
    ext_modules = ext_modules,
    packages = find_packages(),
    )


