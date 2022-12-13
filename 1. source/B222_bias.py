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


outputfolder = '../2. Jmat_loopvals/B222_bias_Jmat/'
path_b222ctab = '../3. Ctabs/B222ctab.csv'
fisherPoints_path = '../3. Ctabs/fisherPoints.csv'
CMASSPoints_path = '../3. Ctabs/CMASS_tri_eff.csv'
LOWZPoints_path = '../3. Ctabs/LOWZ_tri_eff.csv'

# make output folder
if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

# number of fitting functions we are using
nfit = 16

mpfr0 = mpfr(0)

ctab222 = np.loadtxt(path_b222ctab, dtype = int, delimiter = ',')
len222 = len(ctab222)

# function to write the output file name
def outputfile(k1, k2, k3):
	k1_str = str(round(k1,5))
	k2_str = str(round(k2,5))
	k3_str = str(round(k3,5))
	out_filename = outputfolder + 'B222_Jfunc_' + k1_str +'_' + k2_str + '_' + k3_str + '_.h5'
	return out_filename

# utility function to save jmat to h5 file
def saver(outfile, jmat):
	with h5py.File(outfile,'w') as h5_file:
		h5_file.create_dataset("jmat", data= jmat)
	h5_file.close()


def B222jmat(k1,k2,k3):
	
	k1mpf = mpfr(str(k1))
	k2mpf = mpfr(str(k2))
	k3mpf = mpfr(str(k3))

	k12 = k1mpf**2
	k22 = k2mpf**2
	k32 = k3mpf**2

	# clear cache because different triangle
	config.clear_cache()

	Jmat = np.empty((len222, nfit, nfit, nfit),dtype=float)
	for i1 in reversed(range(nfit)):
		for i2 in reversed(range(nfit)):
			for i3 in reversed(range(nfit)):
				for i in range(len222):
					Jmat[i,i1,i2,i3] = J(ctab222[i,0], ctab222[i,1], ctab222[i,2], 
										i1, i2, i3, 
										k12, k22, k32)

	return Jmat

# calculates jmats for a list of ks
def compute_all_B222(triangles):
	for trian in triangles:
		out_filename = outputfile(*trian)
		# only calculates if file still does not exist	
		if not(os.path.isfile(out_filename)):
			start_time = time.time()
			saver(out_filename, B222jmat(*trian))
			end_time = time.time()
			print("--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
	fisher_points = np.loadtxt(LOWZPoints_path, dtype = float, delimiter = ',')
	compute_all_B222(fisher_points)