import numpy as np
import os
import sys
from functools import lru_cache
import pandas as pd

from Jfunc_cython_v4 import computeJ as J
import gmpy2 as gm
from gmpy2 import *
import time

import config
from config import Ltrian_cache, TriaN_cache

gm.get_context().precision = 190
gm.get_context().allow_complex = True

# define in and out folders
ctabfolder = '../3. Ctabs/B3212Redshiftctabks/'
outputfolder = '../2. Jmat_loopvals/B3212Redshift_Jmat_cython/'

if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

filelist = [f for f in os.listdir(ctabfolder) if not f.startswith('.')]
print(filelist)

def get_ks(filename):
	k1 = mpfr(str.split(filename,'_')[1])
	k2 = mpfr(str.split(filename,'_')[2])
	k3 = mpfr(str.split(str.split(filename,'_')[3],'.csv')[0])
	return (k1,k2,k3)

def computeker(i1, k2, ctab_ns, ctab_coefs, Jtriantable):
	mpfr0 = mpfr(0)
	numker = len(ctab_coefs)
	res = 0
	for i in range(numker):
		if ctab_coefs[i] != 0:
			res += ctab_coefs[i]*J(-ctab_ns[i,0], -ctab_ns[i,1], 0, i1, -1, -1, k2, mpfr0, mpfr0)			
	Jtriantable[i1] = res
	return res




def compute_B3212Red_jmat(filename):

	(k1,k2,k3) = get_ks(filename)
	k12 = k1**2
	k22 = k2**2
	k32 = k3**2

	ctab_load = np.loadtxt(ctabfolder + filename, dtype = object, delimiter=',')
	ctab = np.zeros((len(ctab_load),3), dtype = object)

	for i in range(len(ctab)):
		ctab[i, 0] = round(float(ctab_load[i, 0]))
		ctab[i, 1] = round(float(ctab_load[i, 1]))
		ctab[i, 2] = float(str(ctab_load[i, 2]))

	ctab_ns = ctab[:,0:2].astype(int)
	ctab_coefs = ctab[:,2].astype(float)

	Jtriantable = np.empty(16,dtype=float)

	# clear cache because it is a different set of ks 
	config.clear_cache()

	for i1 in range(16):
		print(i1)
		computeker(i1, k12, ctab_ns, ctab_coefs, Jtriantable)
	
	# Output table to csv
	out_filename = outputfolder + 'B3212Redshift_Jfunc_'+ str(float(k1)) + '_' + str(float(k2)) + '_' + str(float(k3)) + '_' +'.csv'
	np.savetxt(out_filename, Jtriantable, delimiter=',')


def compute_all_B3212():
	for file in reversed(filelist):
		(k1,k2,k3) = get_ks(file)
		print(k1,k2,k3)
		out_filename =  outputfolder + 'B3212Redshift_Jfunc_'+ str(float(k1)) + '_' + str(float(k2)) + '_' + str(float(k3)) + '_' +'.csv'
		# if not(os.path.isfile(out_filename)):	
		start_time = time.time()
		compute_B3212Red_jmat(file)
		print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
	compute_all_B3212()