import numpy as np
import os
import sys
from functools import lru_cache
import pandas as pd
import h5py

from Jfunc_cython_v4 import computeJ as J
import gmpy2 as gm
from gmpy2 import mpfr
import time

import config
from config import Ltrian_cache, TriaN_cache

gm.get_context().precision = 190
gm.get_context().allow_complex = True

# load coefficients
# outputfolder = '../2. Jmat_loopvals/B411_bias_Jmat/'
# new jmat output folder to save laptop memory
outputfolder = '/d/jmats/B411/'

path_b411ctab = '../3. Ctabs/B411ctab.csv'
fisherPoints_path = '../3. Ctabs/fisherPoints.csv'
CMASSPoints_path = '../3. Ctabs/CMASS_tri_eff.csv'
LOWZPoints_path = '../3. Ctabs/LOWZ_tri_eff.csv'
basePoints_path = '../3. Ctabs/base_tri_eff.csv'

if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

# number of fitting functions we are using
nfit = 16

ctab411 = np.loadtxt(path_b411ctab, dtype = int, delimiter = ',')
len411 = len(ctab411)

# function to write the output file name
def outputfile(k1, k2, k3):
	k1_str = str(round(k1,5))
	k2_str = str(round(k2,5))
	k3_str = str(round(k3,5))
	out_filename = outputfolder + k1_str +'_' + k2_str + '_' + k3_str + '_.h5'
	return out_filename

# utility function to save jmat to h5 file
def saver(outfile, jmat):
	with h5py.File(outfile,'w') as h5_file:
		h5_file.create_dataset("jmat", data= jmat)
	h5_file.close()


def B411jmat(k1, k2, k3):
	k1mpf = mpfr(str(k1))
	k2mpf = mpfr(str(k2))
	k3mpf = mpfr(str(k3))

	k12 = k1mpf**2
	k22 = k2mpf**2
	k32 = k3mpf**2

	k_cycperms_str = ["k123", "k231", "k312"]

	numperms = len(k_cycperms_str) 

	Jmat = np.empty((len411, nfit, numperms),dtype=float)

	# this dictionary maps permutations in k1, k2, k3 to permutations in (n1,i1), (n2,i2), (n3,i3), represented by (1,2,3)
	# while keeping k1, k2, k3 in the initial ordering
	# we do this so that we can more efficiently use cached values of J
	# here, kijm means the permutation (ki, kj, km)
	k_to_n_perms = {"k123": (1,2,3), "k231": (2,3,1), "k312": (3,1,2)}

	# clear cache because it is a different set of ks 
	config.clear_cache()

	for i1 in reversed(range(nfit)):
		for i in range(len411):
			# iterate over permutations
			n_vec = [ctab411[i,0], ctab411[i,2], ctab411[i,4]]

			# define the index vector (it depends on the ctab too for b411)			
			if ctab411[i,1] == 1:
				i_vec = [i1, -1, -1]
			elif ctab411[i,3] == 1:
				i_vec = [-1, i1, -1]
			elif ctab411[i,5] == 1:
				i_vec = [-1, -1, i1]
			
			for (j, kperm) in enumerate(k_cycperms_str):
				(p1,p2,p3) = k_to_n_perms[kperm]
				Jmat[i, i1, j] =  J(n_vec[p1-1], n_vec[p2-1], n_vec[p3-1],
									i_vec[p1-1], i_vec[p2-1], i_vec[p3-1],
									k12, k22, k32)
	return Jmat

# calculates jmats for a set of triangles
def compute_all_B411(triangles):
	for trian in triangles:
		out_filename = outputfile(*trian)
		# only calculates if file still does not exist	
		if not(os.path.isfile(out_filename)):
			start_time = time.time()
			saver(out_filename, B411jmat(*trian))
			end_time = time.time()
			print("--- %s seconds ---" % (end_time - start_time))
	

if __name__ == "__main__":
	LOWZpoints = np.loadtxt(LOWZPoints_path, dtype = float, delimiter = ',')
	CMASSpoints = np.loadtxt(CMASSPoints_path, dtype = float, delimiter = ',')
	basepoints = np.loadtxt(basePoints_path, dtype = float, delimiter = ',')
	
	compute_all_B411(LOWZpoints)
	print('LOWZ done.')

	compute_all_B411(CMASSpoints)
	print('CMASS done.')
	
	compute_all_B411(basepoints)
	print('Base done.')