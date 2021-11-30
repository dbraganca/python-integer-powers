from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
	name = "babiscython_v4_ubuntu",
	ext_modules = cythonize(
		Extension(
			"babiscython_v4_ubuntu",
			sources = ["babiscython_v4_ubuntu.pyx"],
			include_dirs= [numpy.get_include()],
			# include_dirs = sys.path,
			libraries = ['gmp', 'mpfr', 'mpc'],
		),
		annotate = True),
	install_requires = ["numpy"],
	zip_safe = False
)

# python setup.py build_ext --inplace