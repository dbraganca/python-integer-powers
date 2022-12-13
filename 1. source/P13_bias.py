import numpy as np
import os
import sys
from functools import lru_cache
import pandas as pd
import h5py

from Jfunc_cython_v4 import computeJ as J
import gmpy2 as gm
from gmpy2 import *
import time

import config
from config import Ltrian_cache, TriaN_cache

gm.get_context().precision = 190
gm.get_context().allow_complex = True

# load coefficients
outputfolder = '../2. Jmat_loopvals/P13_bias_Jmat/'
path_p13ctab = '../3. Ctabs/P13ctab.csv'
fisherPoints_path = '../3. Ctabs/fisherPoints.csv'
challengeA_path = '../3. Ctabs/chA_points.csv'
CMASS_path = '../3. Ctabs/CMASS_ks.csv'
LOWZ_path = '../3. Ctabs/LOWZ_ks.csv'

if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

# number of fitting functions we are using
nfit = 16

mpfr0=mpfr(0)

ctab13 = np.loadtxt(path_p13ctab, dtype = int, delimiter = ',')
len13 = len(ctab13)

# function to write the output file name
def outputfile(k1):
	k1_str = str(round(k1,5))
	out_filename = outputfolder + 'P13_Jfunc_' + k1_str + '_.h5'
	return out_filename

# utility function to save jmat to h5 file
def saver(outfile, jmat):
	with h5py.File(outfile,'w') as h5_file:
		h5_file.create_dataset("jmat", data= jmat)
	h5_file.close()

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

# calculates jmats for a set of triangles
def compute_all_P13(ktab):
	for k in ktab:
		out_filename = outputfile(k)
		# only calculates if file still does not exist	
		if not(os.path.isfile(out_filename)):
			start_time = time.time()
			saver(out_filename, P13jmat(k))
			end_time = time.time()
			print("--- %s seconds ---" % (end_time - start_time))
	

if __name__ == "__main__":
	# chA_points = np.loadtxt(challengeA_path, dtype = float, delimiter = ',')
	CMASS_points = np.loadtxt(CMASS_path, dtype = float, delimiter = ',')
	LOWZ_points = np.loadtxt(LOWZ_path, dtype = float, delimiter = ',')
	compute_all_P13(CMASS_points)
	compute_all_P13(LOWZ_points)