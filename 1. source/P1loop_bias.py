import numpy as np
import os
import sys
import h5py
from typing import Callable

from Jfunc_cython_v4 import computeJ as J
import gmpy2 as gm
from gmpy2 import mpfr
import time

import config
from config import Ltrian_cache, TriaN_cache

gm.get_context().precision = 190
gm.get_context().allow_complex = True

diagrams = ['P22','P13']

jmatfolder = '../2. Jmat_loopvals/'
outputfolders = [jmatfolder + diagram + '/' for diagram in diagrams]
P22output, P13output = outputfolders

ctab_folder = '../3. Ctabs/'
path_p22ctab = ctab_folder + 'P22ctab.csv'
path_p13ctab = ctab_folder + 'P13ctab.csv'

# paths for triangles
fisherPoints_path = '../3. Ctabs/fisherPoints.csv'
challengeA_path = '../3. Ctabs/chA_points.csv'
CMASS_path = '../3. Ctabs/CMASS_ks.csv'
LOWZ_path = '../3. Ctabs/LOWZ_ks.csv'
basePoints_path = '../3. Ctabs/base_ks.csv'


# make output folders
for outputfolder in outputfolders:
	if not(os.path.exists(outputfolder)):
		os.makedirs(outputfolder)

# number of fitting functions we are using
nfit = 16

mpfr0 = mpfr(0)

ctab22 = np.loadtxt(path_p22ctab, dtype = int, delimiter = ',')
ctab13 = np.loadtxt(path_p13ctab, dtype = int, delimiter = ',')

len22 = len(ctab22)
len13 = len(ctab13)

# function to write the output file name
def outputfile(k1, outputfolder):
	k1_str = str(round(k1,5))
	out_filename = outputfolder +  k1_str + '_.h5'
	return out_filename

# utility function to save jmat to h5 file
def saver(outfile, jmat):
	with h5py.File(outfile,'w') as h5_file:
		h5_file.create_dataset("jmat", data= jmat)
	h5_file.close()


def P22jmat(k1):
	
	k1mpf = mpfr(str(k1))	
	k12 = k1mpf**2

	# clear cache because different triangle
	config.clear_cache()

	Jmat = np.empty((len22, nfit, nfit),dtype=float)
	for i1 in reversed(range(nfit)):
		for i2 in reversed(range(nfit)):
			for i in range(len22):
					Jmat[i,i1,i2] = J(ctab22[i,0], ctab22[i,1], ctab22[i,2], 
									i1, i2, -1, 
									k12, mpfr0, mpfr0)

	return Jmat

def P13jmat(k1):
	k1mpf = mpfr(str(k1))	
	k12 = k1mpf**2

	Jmat = np.empty((len13, nfit),dtype=float)

	# clear cache because it is a different set of ks 
	config.clear_cache()

	for i1 in reversed(range(nfit)):
		for i in range(len13):
			Jmat[i, i1] =  J(ctab13[i,0], ctab13[i,1], ctab13[i,2],
							 i1, -1, -1,
							k12, mpfr0, mpfr0)
	return Jmat

# calculates jmats for a list of ks
def compute_all(ktab, 
				output_path: str, 
				jmat_fn: Callable[[float] ,np.array]):
	'''
	Calculates a specific jmat_fn for a list of ks and saves it in output_path.
	'''
	for k in ktab:
		out_filename = outputfile(k, output_path)
		# only calculates if file still does not exist	
		if not(os.path.isfile(out_filename)):
			start_time = time.time()
			saver(out_filename, jmat_fn(k))
			end_time = time.time()
			print("--- %.3f seconds ---" % (end_time - start_time))
	return 0

def compute_all_P22(ktab):
	return compute_all(ktab, P22output, P22jmat)

def compute_all_P13(ktab):
	return compute_all(ktab, P13output, P13jmat)

def compute_all_P1loop(ktab):
	compute_all_P22(ktab)
	compute_all_P13(ktab)
	return 0


if __name__ == "__main__":
	# chA_points = np.loadtxt(challengeA_path, dtype = float, delimiter = ',')
	CMASS_points = np.loadtxt(CMASS_path, dtype = float, delimiter = ',')
	LOWZ_points = np.loadtxt(LOWZ_path, dtype = float, delimiter = ',')
	basepoints = np.loadtxt(basePoints_path, dtype = float, delimiter = ',')

	
	compute_all_P1loop(CMASS_points)
	print('CMASS done.')

	compute_all_P1loop(LOWZ_points)
	print('LOWZ done.')

	compute_all_P1loop(basepoints)
	print('Base done.')
