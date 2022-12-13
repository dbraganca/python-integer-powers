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
outputfolder = '../2. Jmat_loopvals/B3212_matter_Jmat/'
ctabfolder = '../3. Ctabs/B3212ctabks/'
# path_b3212ctab = '../3. Ctabs/B3212ctab.csv'

# create output folder if it does not exist
if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

filelist = [f for f in os.listdir(ctabfolder) if not f.startswith('.')]

# number of fitting functions we are using
nfit = 16

def get_ks(filename):
	# filename = 'B3212ctab_' + k1str + '_' + k2str + '_' + k3str + '_.csv'
	k1 = mpfr(str.split(filename,'_')[1])
	k2 = mpfr(str.split(filename,'_')[2])
	k3 = mpfr(str.split(str.split(filename,'_')[3],'.csv')[0])
	return (k1,k2,k3)

# function to write the output file name
def outputfile(k1, k2, k3):
	k1_str = str(round(float(k1),5))
	k2_str = str(round(float(k2),5))
	k3_str = str(round(float(k3),5))
	# out_filename = outputfolder + 'B3212_Jfunc_' + k1_str +'_' + k2_str + '_' + k3_str + '_.h5'
	out_filename = outputfolder + 'B3212_Jfunc_' + k1_str +'_' + k2_str + '_' + k3_str + '_.csv'
	return out_filename

# utility function to save jmat to h5 file
def saver(outfile, jmat):
	# with h5py.File(outfile,'w') as h5_file:
	# 	h5_file.create_dataset("jmat", data= jmat)
	# h5_file.close()
	np.savetxt(outfile, jmat, delimiter=',')

# function to load ctab and separate into exponents and coefficients
def load_ctab_b3212(filename):
	ctab = np.loadtxt(ctabfolder + filename, dtype = object, delimiter = ',')
	ctab_ns = np.around(ctab[:,0:3].astype(float)).astype(int)
	ctab_coefs = ctab[:,3:].astype(str).astype(float)
	return ctab_ns, ctab_coefs


def B3212jmat(k1, k2, k3, ctab_ns):
	k1mpf = mpfr(str(k1))
	k2mpf = mpfr(str(k2))
	k3mpf = mpfr(str(k3))

	k12 = k1mpf**2
	k22 = k2mpf**2
	k32 = k3mpf**2

	k_perms_str = ["k123", "k132", "k213",
				   "k231", "k312", "k321"]

	numperms = len(k_perms_str) 
	len3212 = len(ctab_ns)

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
			n_vec = [ctab_ns[i,0], ctab_ns[i,1], ctab_ns[i,2]]
			i_vec = [i1, -1, -1]
			for (j, kperm) in enumerate(k_perms_str):
				(p1,p2,p3) = k_to_n_perms[kperm]
				Jmat[i, i1, j] =  J(n_vec[p1-1], n_vec[p2-1], n_vec[p3-1],
										i_vec[p1-1], i_vec[p2-1], i_vec[p3-1],
										k12, k22, k32)
	return Jmat

def compute_B3212(filename):
	(k1,k2,k3) = get_ks(filename)
	ctab_ns, ctab_coefs = load_ctab_b3212(filename)

	# calculate jmat for all permutations without coefficients
	jmat = B3212jmat(k1, k2, k3, ctab_ns)

	# combine with coefficients
	# jmatcoefs is a 2-d array with dimensions nfit x 6
	jmatcoef = np.einsum('ijk,ik->jk', jmat, ctab_coefs)
	return jmatcoef


# calculates jmats for filelist
def compute_all_B3212():
	for file in filelist:
		(k1,k2,k3) = get_ks(file)
		out_filename = outputfile(k1,k2,k3)
		# only calculates if file still does not exist
		if not(os.path.isfile(out_filename)) and k1 != k2:
			start_time = time.time()
			saver(out_filename, compute_B3212(file))
			end_time = time.time()
			print("--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
	compute_all_B3212()