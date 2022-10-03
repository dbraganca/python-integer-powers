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
ctabfolder = '../3. Ctabs/B3211Redshiftctabks/'
outputfolder = '../2. Jmat_loopvals/B3211Redshift_Jmat_cython/'

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


def computeker(i1,i2, k12, k22, k32, ctab_ns, ctab_coefs, Jtriantable):
	numker = len(ctab_coefs)
	res = 0
	for i in range(numker):
		if ctab_coefs[i] != 0:
			term = ctab_coefs[i]*J(-ctab_ns[i,0], -ctab_ns[i,1], -ctab_ns[i,2], i1, -1, i2, k12, k22, k32)
			res += term
	Jtriantable[i1,i2] = res
	return res

def compute_B3211_jmat(filename):

	(k1,k2,k3) = get_ks(filename)
	k12 = k1**2
	k22 = k2**2
	k32 = k3**2

	ctab_load = np.loadtxt(ctabfolder + filename, dtype = object)
	numker = len(ctab_load)
	ctab = np.zeros((numker,4), dtype = object)

	for i in range(numker):
		ctab[i, 0] = round(float(ctab_load[i, 0]))
		ctab[i, 1] = round(float(ctab_load[i, 1]))
		ctab[i, 2] = round(float(ctab_load[i, 2]))
		ctab[i, 3] = float(str(ctab_load[i, 3]))

	ctab_ns = ctab[:,0:3].astype(int)
	ctab_coefs = ctab[:,3].astype(float)

	# clear cache because it is a different set of ks 
	config.clear_cache()

	Jtriantable = np.empty((16,16),dtype=float)

	for i1 in reversed(range(16)):
		for i2 in reversed(range(16)):
			print(i2,i1)			
			computeker(i1,i2, k12, k22, k32, ctab_ns, ctab_coefs, Jtriantable)

	# Output the table to csv 
	out_df = pd.DataFrame(Jtriantable, dtype = object)
	out_filename = outputfolder + 'B3211Redshift_Jfunc_'+str(float(k1))+'_' + str(float(k2)) + '_' + str(float(k3)) + '_' +'.csv'
	out_df.to_csv(out_filename,index = False)


def compute_all_B3211():
	for file in reversed(filelist):
		(k1,k2,k3)=get_ks(file)
	
		out_filename = outputfolder + 'B3211Redshift_Jfunc_'+str(float(k1))+'_' + str(float(k2)) + '_' + str(float(k3)) + '_' +'.csv'
		if k1==k2 and k2==k3:
			if not(os.path.isfile(out_filename)):	
				start_time = time.time()
				compute_B3211_jmat(file)
				print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
	compute_all_B3211()