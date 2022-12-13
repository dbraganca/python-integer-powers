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

# define paths
outputfolder = '../2. Jmat_loopvals/B3212_bias_Jmat/'
path_b3212ctab = '../3. Ctabs/B3212ctab.csv'
fisherPoints_path = '../3. Ctabs/fisherPoints.csv'
CMASSPoints_path = '../3. Ctabs/CMASS_tri_eff.csv'
LOWZPoints_path = '../3. Ctabs/LOWZ_tri_eff.csv'

# create output folder if it does not exist
if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

# number of fitting functions we are using
nfit = 16

ctab3212 = np.loadtxt(path_b3212ctab, dtype = int, delimiter = ',')
len3212 = len(ctab3212)

# function to write the output file name
def outputfile(k1, k2, k3):
	k1_str = str(round(k1,5))
	k2_str = str(round(k2,5))
	k3_str = str(round(k3,5))
	out_filename = outputfolder + 'B3212_Jfunc_' + k1_str +'_' + k2_str + '_' + k3_str + '_.h5'
	return out_filename

# utility function to save jmat to h5 file
def saver(outfile, jmat):
	with h5py.File(outfile,'w') as h5_file:
		h5_file.create_dataset("jmat", data= jmat)
	h5_file.close()

def B3212jmat(k1, k2, k3):
	k1mpf = mpfr(str(k1))
	k2mpf = mpfr(str(k2))
	k3mpf = mpfr(str(k3))

	k12 = k1mpf**2
	k22 = k2mpf**2
	k32 = k3mpf**2

	k_perms_str = ["k123", "k132", "k213",
				   "k231", "k312", "k321"]

	numperms = len(k_perms_str) 

	Jmat = np.empty((len3212, nfit, numperms),dtype=float)

	# this dictionary maps permutations in k1, k2, k3 to permutations in (n1,i1), (n2,i2), (n3,i3), represented by (1,2,3)
	# while keeping k1, k2, k3 in the initial ordering
	# we do this so that we can more efficiently use cached values of J
	# here, kijm means the permutation (ki, kj, km)
	k_to_n_perms = {"k123": (1,2,3), "k132": (2,1,3), "k213": (1,3,2),
					"k231": (2,3,1), "k312": (3,1,2), "k321": (3,2,1)}

	# clear cache because it is a different set of ks 
	config.clear_cache()

	for i1 in reversed(range(16)):
		for i in range(len3212):
			# iterate over permutations
			n_vec = [ctab3212[i,0], ctab3212[i,1], ctab3212[i,2]]
			i_vec = [i1, -1, -1]
			for (j, kperm) in enumerate(k_perms_str):
				(p1,p2,p3) = k_to_n_perms[kperm]
				Jmat[i, i1, j] =  J(n_vec[p1-1], n_vec[p2-1], n_vec[p3-1],
										i_vec[p1-1], i_vec[p2-1], i_vec[p3-1],
										k12, k22, k32)
	return Jmat


# calculates jmats for a set of triangles
def compute_all_B3212(triangles):
	for trian in triangles:
		out_filename = outputfile(*trian)
		# only calculates if file still does not exist	
		if not(os.path.isfile(out_filename)):
			start_time = time.time()
			saver(out_filename, B3212jmat(*trian))
			end_time = time.time()
			print("--- %s seconds ---" % (end_time - start_time))
	

if __name__ == "__main__":
	# fisher_points = np.loadtxt(CMASSPoints_path, dtype = float, delimiter = ',')
	fisher_points = np.loadtxt(LOWZPoints_path, dtype = float, delimiter = ',')
	compute_all_B3212(fisher_points)