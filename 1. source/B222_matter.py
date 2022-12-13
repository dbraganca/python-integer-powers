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


outputfolder = '../2. Jmat_loopvals/B222_matter_Jmat/'
ctabfolder = '../3. Ctabs/B222ctabks/'

# make output folder
if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

# number of fitting functions we are using
nfit = 16

filelist = [f for f in os.listdir(ctabfolder) if not f.startswith('.')]

mpfr0 = mpfr(0)

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
	out_filename = outputfolder + 'B222_Jfunc_' + k1_str +'_' + k2_str + '_' + k3_str + '_.h5'
	return out_filename

# utility function to save jmat to h5 file
def saver(outfile, jmat):
	with h5py.File(outfile,'w') as h5_file:
		h5_file.create_dataset("jmat", data= jmat)
	h5_file.close()

# function to load ctab and separate into exponents and coefficients
def load_ctab(filename):
	ctab = np.loadtxt(ctabfolder + filename, dtype = object, delimiter = ',')
	ctab_ns = np.around(ctab[:,0:3].astype(float)).astype(int)
	ctab_coefs = (ctab[:,3:].astype(str)
							.astype(float))
	return ctab_ns, ctab_coefs



def B222jmat(k1,k2,k3,ctab_ns):
	
	k1mpf = mpfr(str(k1))
	k2mpf = mpfr(str(k2))
	k3mpf = mpfr(str(k3))

	k12 = k1mpf**2
	k22 = k2mpf**2
	k32 = k3mpf**2

	len222 = len(ctab_ns)

	# clear cache because different triangle
	config.clear_cache()

	Jmat = np.empty((len222, nfit, nfit, nfit),dtype=float)
	for i1 in reversed(range(nfit)):
		for i2 in reversed(range(nfit)):
			for i3 in reversed(range(nfit)):
				for i in range(len222):
					Jmat[i,i1,i2,i3] = J(ctab_ns[i,0], ctab_ns[i,1], ctab_ns[i,2], 
										i1, i2, i3, 
										k12, k22, k32)

	return Jmat

def compute_B222(filename):
	(k1,k2,k3) = get_ks(filename)
	ctab_ns, ctab_coefs = load_ctab(filename)

	# calculate jmat for all permutations without coefficients
	jmat = B222jmat(k1, k2, k3, ctab_ns)

	# combine with coefficients
	# jmatcoefs is a 3-d array with dimensions nfit x nfit x nfit
	jmatcoef = np.einsum('ijkl,i->jkl', jmat, ctab_coefs.reshape(-1))
	return jmatcoef



# calculates jmats for a list of ks
def compute_all_B222():
	for file in filelist:
		(k1,k2,k3) = get_ks(file)
		out_filename = outputfile(k1,k2,k3)
		# only calculates if file still does not exist
		# if not(os.path.isfile(out_filename)) and k1 != k2:
		if k1!=k2:
			start_time = time.time()
			saver(out_filename, compute_B222(file))
			end_time = time.time()
			print("--- %s seconds ---" % (end_time - start_time))

	
if __name__ == "__main__":
	compute_all_B222()