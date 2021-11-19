from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
	name = "B222_jfunc_cython_v1",
	ext_modules = cythonize(
		Extension(
			"B222_jfunc_cython_v1",
			sources = ["B222_jfunc_cython_v1.pyx"],
			include_dirs= [numpy.get_include()],
			# include_dirs = sys.path,
			libraries = ['gmp', 'mpfr', 'mpc'],
		),
		annotate = True),
	install_requires = ["numpy"],
	zip_safe = False
)

# python setup.py build_ext --inplace