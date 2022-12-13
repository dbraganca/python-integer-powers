import numpy as np
import os
import sys
import h5py

from Jfunc_cython_v4 import computeJ as J
import gmpy2 as gm
from gmpy2 import *
import time

import config
from config import Ltrian_cache, TriaN_cache

gm.get_context().precision = 190
gm.get_context().allow_complex = True


outputfolder = '../2. Jmat_loopvals/P22_bias_Jmat/'
path_p22ctab = '../3. Ctabs/P22ctab.csv'
fisherPoints_path = '../3. Ctabs/fisherPoints.csv'
challengeA_path = '../3. Ctabs/chA_points.csv'
CMASS_path = '../3. Ctabs/CMASS_ks.csv'
LOWZ_path = '../3. Ctabs/LOWZ_ks.csv'



# make output folder
if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

# number of fitting functions we are using
nfit = 16

mpfr0 = mpfr(0)

ctab22 = np.loadtxt(path_p22ctab, dtype = int, delimiter = ',')
len22 = len(ctab22)

# function to write the output file name
def outputfile(k1):
	k1_str = str(round(k1,5))
	out_filename = outputfolder + 'P22_Jfunc_' + k1_str + '_.h5'
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

# calculates jmats for a list of ks
def compute_all_P22(ktab):
	for k in ktab:
		out_filename = outputfile(k)
		# only calculates if file still does not exist	
		if not(os.path.isfile(out_filename)):
			start_time = time.time()
			saver(out_filename, P22jmat(k))
			end_time = time.time()
			print("--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
	# chA_points = np.loadtxt(challengeA_path, dtype = float, delimiter = ',')
	CMASS_points = np.loadtxt(CMASS_path, dtype = float, delimiter = ',')
	LOWZ_points = np.loadtxt(LOWZ_path, dtype = float, delimiter = ',')
	compute_all_P22(CMASS_points)
	compute_all_P22(LOWZ_points)