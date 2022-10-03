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

# load coefficients
ctabfolder = '../3. Ctabs/B411Redshiftctabks/'
outputfolder = '../2. Jmat_loopvals/B411Redshift_Jmat_cython/'

if not(os.path.exists(outputfolder)):
	os.makedirs(outputfolder)

filelist = [f for f in os.listdir(ctabfolder) if not f.startswith('.')]

def get_ks(filename):
	# filename = 'B411ctab_' + k1str + '_' + k2str + '_' + k3str + '_.csv'
	k1 = mpfr(str.split(filename,'_')[1])
	k2 = mpfr(str.split(filename,'_')[2])
	k3 = mpfr(str.split(str.split(filename,'_')[3],'.csv')[0])
	return (k1,k2,k3)

def computeker(i1, k12, k22, k32, ctab_ns, ctab_coefs, Jtriantable):
	# ctab_ns has the form (n1, i, n2, j, n3, k) where ijk is 100, 010 or 001 
	# depending on the change of variable needed to make
	numker = len(ctab_coefs)
	res = 0
	for i in range(numker):
		if ctab_coefs[i] != 0:
			if ctab_ns[i, 1] != 0:
				# no change of variable: Plin inside integral is P(q)
				term = ctab_coefs[i] * J(-ctab_ns[i,0], -ctab_ns[i,2], -ctab_ns[i,4], i1, -1, -1, k12, k22, k32)

			elif ctab_ns[i, 3] != 0:
				# change of variable q -> k1 - q
				# Plin inside integral is P(k1 - q)
				term = ctab_coefs[i] * J(-ctab_ns[i,0], -ctab_ns[i,2], -ctab_ns[i,4], -1, i1, -1, k12, k22, k32)

			elif ctab_ns[i, 5] != 0:
				# change of variable q -> k2 + q
				# Plin inside integral is P(k2 + q)
				term = ctab_coefs[i] * J(-ctab_ns[i,0], -ctab_ns[i,2], -ctab_ns[i,4], -1, -1, i1, k12, k22, k32)
			else:
				print(i,"ERROR in computeker: case not considered")
			res += term

	Jtriantable[i1] = res
	return res


def compute_B411_jmat(filename):

	(k1,k2,k3) = get_ks(filename)
	k12 = k1**2
	k22 = k2**2
	k32 = k3**2

	ctab_load = np.loadtxt(ctabfolder + filename, 
							dtype = object, 
							delimiter=',')
	numker = len(ctab_load)
	ctab = np.zeros((numker,7), dtype = object)

	for i in range(len(ctab)):
		ctab[i, 0] = round(float(ctab_load[i, 0]))
		ctab[i, 1] = round(float(ctab_load[i, 1]))
		ctab[i, 2] = round(float(ctab_load[i, 2]))
		ctab[i, 3] = round(float(ctab_load[i, 3]))
		ctab[i, 4] = round(float(ctab_load[i, 4]))
		ctab[i, 5] = round(float(ctab_load[i, 5]))
		ctab[i, 6] = float(str(ctab_load[i, 6]))

	ctab_ns = ctab[:,0:6].astype(int)
	ctab_coefs = ctab[:,6].astype(float)

	Jtriantable = np.empty((16,),dtype=float)

	# clear cache because it is a different set of ks 
	config.clear_cache()

	for i1 in range(16):
		print(i1)
		computeker(i1, k12, k22, k32, ctab_ns, ctab_coefs, Jtriantable)

	out_df = pd.DataFrame(Jtriantable, dtype = object)
	print(k1,k2,k3)
	out_filename = outputfolder + 'B411Redshift_Jfunc_'+str(float(k1))+'_' + str(float(k2)) + '_' + str(float(k3)) + '_' +'.csv'
	out_df.to_csv(out_filename, index = False)
	
def compute_all_B411():
	for file in filelist:
		(k1,k2,k3)=get_ks(file)
		out_filename = outputfolder + 'B411Redshift_Jfunc_'+str(float(k1))+'_' + str(float(k2)) + '_' + str(float(k3)) + '_' +'.csv'
		if not(os.path.isfile(out_filename)):	
			if k1==k2 and k2==k3:
				start_time = time.time()
				compute_B411_jmat(file)
				end_time = time.time()
				print("--- %s seconds ---" % (end_time - start_time))

if __name__ == "__main__":
	compute_all_B411()