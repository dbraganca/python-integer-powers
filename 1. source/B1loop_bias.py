import numpy as np
import os
import h5py
from typing import Callable

from Jfunc_cython_v4 import computeJ as J
import gmpy2 as gm
from gmpy2 import mpfr
import time

import config
from config import Ltrian_cache, TriaN_cache

gm.get_context().precision = 190
gm.get_context().allow_complex = True

# outputfolder = '../2. Jmat_loopvals/B222/'

# new jmat output folder to save laptop memory
diagrams = ['B222','B3211','B3212','B411']

# make folders to store jmats
jmatfolder = '/d/jmats/'
outputfolders = [jmatfolder + diagram + '/' for diagram in diagrams]
B222output, B3211output, B3212output, B411output = outputfolders

# paths where ctabs are stored
ctab_folder = '../3. Ctabs/'
path_b222ctab = ctab_folder + 'B222ctab.csv'
path_b3211ctab = ctab_folder + 'B3211ctab.csv'
path_b3212ctab = ctab_folder + 'B3212ctab.csv'
path_b411ctab = ctab_folder + 'B411ctab.csv'

# paths for triangles
CMASSPoints_path = '../3. Ctabs/CMASS_tri_eff.csv'
LOWZPoints_path = '../3. Ctabs/LOWZ_tri_eff.csv'
basePoints_path = '../3. Ctabs/base_tri_eff.csv'
BOSSPoints_path = '../3. Ctabs/BOSS_tri_eff.csv'

# make output folders
for outputfolder in outputfolders:
	if not(os.path.exists(outputfolder)):
		os.makedirs(outputfolder)

# number of fitting functions we are using
nfit = 16

mpfr0 = mpfr(0)

ctab222 = np.loadtxt(path_b222ctab, dtype = int, delimiter = ',')
ctab3211 = np.loadtxt(path_b3211ctab, dtype = int, delimiter = ',')
ctab3212 = np.loadtxt(path_b3212ctab, dtype = int, delimiter = ',')
ctab411 = np.loadtxt(path_b411ctab, dtype = int, delimiter = ',')

len222 = len(ctab222)
len3211 = len(ctab3211)
len3212 = len(ctab3212)
len411 = len(ctab411)

# function to write the output file name
def outputfile(k1, k2, k3, outputfolder):
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


def B222jmat(k1,k2,k3):
	
	k1mpf = mpfr(str(k1))
	k2mpf = mpfr(str(k2))
	k3mpf = mpfr(str(k3))

	k12 = k1mpf**2
	k22 = k2mpf**2
	k32 = k3mpf**2

	# clear cache because different triangle
	config.clear_cache()

	Jmat = np.empty((len222, nfit, nfit, nfit),dtype=float)
	for i1 in reversed(range(nfit)):
		for i2 in reversed(range(nfit)):
			for i3 in reversed(range(nfit)):
				for i in range(len222):
					Jmat[i,i1,i2,i3] = J(ctab222[i,0], ctab222[i,1], ctab222[i,2], 
										i1, i2, i3, 
										k12, k22, k32)

	return Jmat

def B3211jmat(k1, k2, k3):
	k1mpf = mpfr(str(k1))
	k2mpf = mpfr(str(k2))
	k3mpf = mpfr(str(k3))

	k12 = k1mpf**2
	k22 = k2mpf**2
	k32 = k3mpf**2

	k_perms_str = ["k123", "k132", "k213",
				   "k231", "k312", "k321"]

	numperms = len(k_perms_str) 

	Jmat = np.empty((len3211, nfit, nfit, numperms),dtype=float)

	# this dictionary maps permutations in k1, k2, k3 to permutations in (n1,i1), (n2,i2), (n3,i3), represented by (1,2,3)
	# while keeping k1, k2, k3 in the initial ordering
	# we do this so that we can more efficiently use cached values of J
	# here, kijm means the permutation (ki, kj, km)
	k_to_n_perms = {"k123": (1,2,3), "k132": (2,1,3), "k213": (1,3,2),
					"k231": (2,3,1), "k312": (3,1,2), "k321": (3,2,1)}

	# clear cache because it is a different set of ks 
	config.clear_cache()

	for i1 in reversed(range(nfit)):
		for i2 in reversed(range(nfit)):
			for i in range(len3211):

				# iterate over permutations
				n_vec = [ctab3211[i,0], ctab3211[i,1], ctab3211[i,2]]
				i_vec = [i1, -1, i2]

				for (j, kperm) in enumerate(k_perms_str):
					(p1,p2,p3) = k_to_n_perms[kperm]
					Jmat[i, i1, i2, j] =  J(n_vec[p1-1], n_vec[p2-1], n_vec[p3-1],
												   i_vec[p1-1], i_vec[p2-1], i_vec[p3-1],
												   k12, k22, k32)
	return Jmat

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

def compute_all(triangles, 
				output_path: str, 
				jmat_fn: Callable[[float, float, float],np.array]):
	'''
	Calculates a specific jmat_fn for a list of triangles and saves it in output_path.
	'''
	for trian in triangles:
		out_filename = outputfile(trian[0], trian[1], trian[2], output_path)
		# only calculates if file still does not exist	
		if not(os.path.isfile(out_filename)):
			start_time = time.time()
			saver(out_filename, jmat_fn(*trian))
			end_time = time.time()
			print("--- %.2f seconds ---" % (end_time - start_time))
	return 0

def compute_all_B222(triangles):
	return compute_all(triangles, B222output, B222jmat)

def compute_all_B3211(triangles):
	return compute_all(triangles, B3211output, B3211jmat)

def compute_all_B3212(triangles):
	return compute_all(triangles, B3212output, B3212jmat)

def compute_all_B411(triangles):
	return compute_all(triangles, B411output, B411jmat)

def compute_all_B1loop(triangles):
	compute_all_B222(triangles)
	compute_all_B3211(triangles)
	compute_all_B3212(triangles)
	compute_all_B411(triangles)
	return 0

if __name__ == "__main__":
	LOWZpoints = np.loadtxt(LOWZPoints_path, dtype = float, delimiter = ',')
	CMASSpoints = np.loadtxt(CMASSPoints_path, dtype = float, delimiter = ',')
	basepoints = np.loadtxt(basePoints_path, dtype = float, delimiter = ',')
	BOSSpoints = np.loadtxt(BOSSPoints_path, dtype = float, delimiter = ',')


	compute_all_B1loop(LOWZpoints)
	print('LOWZ done.')

	compute_all_B1loop(CMASSpoints)
	print('CMASS done.')

	compute_all_B1loop(basepoints)
	print('Base done.')

	compute_all_B1loop(BOSSpoints)
	print('BOSS done.')