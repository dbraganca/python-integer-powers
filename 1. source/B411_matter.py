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
outputfolder = '../2. Jmat_loopvals/B411_matter_Jmat/'
ctabfolder = '../3. Ctabs/B411ctabks/'

path_b411ctab = '../3. Ctabs/B411ctabks/B411ctab_matter.dat'
scalene_triangles_path = '../3. Ctabs/matter_scalene_triangles.dat'

if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

# number of fitting functions we are using
nfit = 16
ctab411 = np.loadtxt(path_b411ctab, dtype = int)
len411 = len(ctab411)

def get_ks(filename):
	# filename = 'B222ctab_' + k1str + '_' + k2str + '_' + k3str + '_.csv'
	k1 = mpfr(str.split(filename,'_')[1])
	k2 = mpfr(str.split(filename,'_')[2])
	k3 = mpfr(str.split(str.split(filename,'_')[3],'.csv')[0])
	return (k1,k2,k3)

# function to write the output file name
def outputfile(k1, k2, k3):
	k1_str = str(round(float(k1),5))
	k2_str = str(round(float(k2),5))
	k3_str = str(round(float(k3),5))
	out_filename = outputfolder + 'B411_Jfunc_' + k1_str +'_' + k2_str + '_' + k3_str + '_.h5'
	return out_filename

# utility function to save jmat to h5 file
def saver(outfile, jmat):
	with h5py.File(outfile,'w') as h5_file:
		h5_file.create_dataset("jmat", data= jmat)
	h5_file.close()


# function to load ctab and separate into exponents and coefficients
def load_ctab_b411(filename):
	ctab = np.loadtxt(ctabfolder + filename, dtype = object)
	ctab_ns = np.around(ctab[:,0:6].astype(float)).astype(int)
	ctab_coefs = (ctab[:,6:].astype(str)
							.astype(float))
	return ctab_ns, ctab_coefs


def B411jmat(k1, k2, k3):
	k1mpf = mpfr(str(k1))
	k2mpf = mpfr(str(k2))
	k3mpf = mpfr(str(k3))

	k12 = k1mpf**2
	k22 = k2mpf**2
	k32 = k3mpf**2

	k_cycperms_str = ["k123", "k231", "k312"]

	numperms = len(k_cycperms_str) 
	# len411 = len(ctab411)

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

# function to perform contraction here - not necessary
def compute_B411(filename):
	(k1,k2,k3) = get_ks(filename)
	ctab_ns, ctab_coefs = load_ctab_b411(filename)

	# calculate jmat for all permutations without coefficients
	jmat = B411jmat(k1, k2, k3, ctab_ns)

	# combine with coefficients
	# jmatcoefs is a 3-d array with dimensions nfit x nfit x nfit
	jmatcoef = np.einsum('ijk,ik->jk', jmat, ctab_coefs)
	return jmatcoef


# calculates jmats for a set of triangles
def compute_all_B411(triangles):
	for trian in triangles:
		out_filename = outputfile(*trian)
		# only calculates if file still does not exist	
		# if not(os.path.isfile(out_filename)):
		start_time = time.time()
		saver(out_filename, B411jmat(*trian))
		end_time = time.time()
		print("--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
	scalene_tris = np.loadtxt(scalene_triangles_path, dtype = float)
	compute_all_B411(scalene_tris)