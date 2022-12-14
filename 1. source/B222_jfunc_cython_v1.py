import numpy as np
import os
from functools import lru_cache
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
ctabfolder = '../3. Ctabs/B222ctabks/'
outputfolder = '../2. Jmat_loopvals/B222_Jmat_cython/'

# make output folder
if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

filelist = [f for f in os.listdir(ctabfolder) if not f.startswith('.')]

def get_ks(filename):
	# filename = 'B222ctab_' + k1str + '_' + k2str + '_' + k3str + '_.csv'
	k1 = mpfr(str.split(filename,'_')[1])
	k2 = mpfr(str.split(filename,'_')[2])
	k3 = mpfr(str.split(str.split(filename,'_')[3],'.csv')[0])
	return (k1,k2,k3)

def computeker(i1,i2,i3, k12, k22, k32, ctab_ns, ctab_coefs, Jtriantable):
	numker = len(ctab_coefs)
	res = 0
	for i in range(numker):
		if ctab_coefs[i] != 0:
			term = ctab_coefs[i]*J(-ctab_ns[i,0], -ctab_ns[i,1], -ctab_ns[i,2], i1, i2, i3, k12, k22, k32)
			res += term
	Jtriantable[i1,i2,i3] = res
	return res

# function to load ctab and separate into exponents and coefficients
def load_ctab(filename):
	ctab = np.loadtxt(ctabfolder + filename, dtype = object)
	ctab_ns = np.around(ctab[:,0:3].astype(float)).astype(int)
	ctab_coefs = (ctab[:,3:].astype(str)
							.astype(float))
	return ctab_ns, ctab_coefs

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

def compute_B222_jmat(filename):
	
	(k1,k2,k3) = get_ks(filename)
	k12 = k1**2
	k22 = k2**2
	k32 = k3**2

	ctab_ns, ctab_coefs = load_ctab(filename)

	Jtriantable = np.empty((16,16,16),dtype=float)

	# clear cache because it is a different set of ks 
	config.clear_cache()

	for i1 in reversed(range(16)):
		for i2 in reversed(range(16)):
			print(i2,i1)
			for i3 in reversed(range(16)):				
				computeker(i1, i2, i3, k12, k22, k32, ctab_ns, ctab_coefs, Jtriantable)
	
	# Output the table to h5 
	out_filename = outputfile(k1,k2,k3)
	saver(out_filename, Jtriantable)

def compute_all_B222():
	for file in filelist:
		(k1,k2,k3)=get_ks(file)		
		out_filename = outputfile(k1,k2,k3)
		if not(os.path.isfile(out_filename)):	
			print(float(k1), float(k2), float(k3))
			start_time = time.time()
			compute_B222_jmat(file)
			print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
	compute_all_B222()