import numpy as np
import os
import sys
from functools import lru_cache
import pandas as pd

from Jfunc_cython_v4 import computeJ as J

import gmpy2 as gm
from gmpy2 import *
import time

from config import Ltrian_cache, TriaN_cache

gm.get_context().precision = 190
gm.get_context().allow_complex = True

# define in and out folders
ctabfolder = '../3. Ctabs/P22ctabks/'
outputfolder = '../2. Jmat_loopvals/P22_Jmat_cython/'

if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

filelist = [f for f in os.listdir(ctabfolder) if not f.startswith('.')]

def computeker(i1, i2, k2, ctab_ns, ctab_coefs, Jtriantable):
	mpfr0 = mpfr(0)
	numker = len(ctab_coefs)
	res = 0
	for i in range(numker):
		if ctab_coefs[i] != 0:
			res += ctab_coefs[i]*J(-ctab_ns[i,0], -ctab_ns[i,1], 0, i1, i2, -1, k2, mpfr0, mpfr0)			
	Jtriantable[i1,i2] = res
	return res

def get_ks(filename):
	# filename = 'P22ctab_' + k1str + '_.csv'
	k1 = mpfr(str.split(filename,'_')[1])
	return k1

def compute_P22_jmat(filename):

	k1 = get_ks(filename)
	k12 = k1**2

	ctab_load = np.loadtxt(ctabfolder + filename, dtype = object)
	ctab = np.zeros((len(ctab_load),3), dtype = object)

	for i in range(len(ctab)):
		ctab[i, 0] = round(float(ctab_load[i, 0]))
		ctab[i, 1] = round(float(ctab_load[i, 1]))
		ctab[i, 2] = float(str(ctab_load[i, 2]))

	ctab_ns = ctab[:,0:2].astype(int)
	ctab_coefs = ctab[:,2].astype(float)

	Jtriantable = np.empty((16,16),dtype=float)

	# clear cache
	Ltrian_cache.clear()
	TriaN_cache.clear()

	for i1 in range(16):
		print(i1)
		for i2 in range(16):
			computeker(i1,i2, k12, ctab_ns, ctab_coefs, Jtriantable)

	# Output table to csv
	out_filename =  outputfolder + 'P22_Jfunc_'+ str(float(k1)) + '_' +'.csv'
	np.savetxt(out_filename, Jtriantable, delimiter=',')

# loop over files in folder
def compute_all_P22():
	for file in reversed(filelist):
		k1 = get_ks(file)
		out_filename =  outputfolder + 'P22_Jfunc_'+ str(float(k1)) + '_' +'.csv'
		if not(os.path.isfile(out_filename)):
			start_time = time.time()
			compute_P22_jmat(file)
			end_time = time.time()
			print("--- %s seconds ---" % (end_time - start_time))

if __name__ == "__main__":
	compute_all_P22()