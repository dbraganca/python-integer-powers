from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
	name = "Jfunc_cython_v4",
	ext_modules = cythonize(
		Extension(
			"Jfunc_cython_v4",
			sources = ["Jfunc_cython_v4.pyx"],
			include_dirs= [numpy.get_include()],
			# include_dirs = sys.path,
			libraries = ['gmp', 'mpfr', 'mpc'],
			#extra_compile_args=['-fopenmp'],
        	#extra_link_args=['-fopenmp'],
		),
		annotate = True),
	install_requires = ["numpy"],
	zip_safe = False
)

# python setup.py build_ext --inplace