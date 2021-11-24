import cython
cimport cython
import numpy as np
cimport numpy as np
import gmpy2 as gm
from gmpy2 import atan, log, sqrt, gamma, acos
from gmpy2 cimport *
import os
import sys
from functools import lru_cache
from babiscython_v4_ubuntu import Ltrian
#from babiscython_v4_ubuntu import Ltrian_complex
import time

cdef extern from "complex.h":
	double complex conj(double complex z)

cdef extern from "gmp.h":
	void mpz_set_si(mpz_t, long)

cdef extern from "mpfr.h":
	void mpfr_init2(mpfr_t, mpfr_prec_t)

	int mpfr_set_d(mpfr_t, double, mpfr_rnd_t)

	int mpfr_const_pi (mpfr_t, mpfr_rnd_t)

cdef extern from "mpc.h":

	mpc_t mpc_cos(mpc_t, mpc_t, mpc_rnd_t)

	void mpc_init2(mpc_t, mpfr_prec_t)

	void mpc_clear(mpc_t)

	int mpc_set_d_d(mpc_t, double, double, mpc_rnd_t)


import_gmpy2()

gm.get_context().precision = 190
gm.get_context().allow_complex = True

cdef long double PI = acos(-1)
cdef long double SQRT_PI = sqrt(PI)
cdef mpfr kpeak1 = mpfr(str(-0.034))
cdef mpfr kpeak2 = mpfr(str(-0.001))
cdef mpfr kpeak3 = mpfr(str(-0.000076))
cdef mpfr kpeak4 = mpfr(str(-0.0000156))
cdef mpfr kuv1 = mpfr(str(0.069))
cdef mpfr kuv2 = mpfr(str(0.0082))
cdef mpfr kuv3 = mpfr(str(0.0013))
cdef mpfr kuv4 = mpfr(str(0.0000135))

matarray_val = np.reshape(np.fromfile('matdiogotobabis', np.complex128),(16,33))
cdef double complex[:,:] matdiogotobabisnowig = matarray_val


#cdef double complex[:] matarray = matarray_val
#cdef double complex[:,:] matdiogotobabisnowig = np.reshape(matarray,(16, 33))

cdef long lenfbabis = 33
cdef long lenfdiogo = 16

fbabisparamtab = np.zeros((lenfbabis,3),dtype=object)
fbabisparamtab[0]=[ mpc(0), 0, 0]
fbabisparamtab[1]=[ -kpeak2 - kuv2*1j, 1, 1]
fbabisparamtab[2]=[ -kpeak2 + kuv2*1j, 1, 1]
fbabisparamtab[3]=[ -kpeak3 - 1j*kuv3, 0, 1]
fbabisparamtab[4]=[ -kpeak3 + 1j*kuv3, 0, 1]
fbabisparamtab[5]=[ -kpeak3 - 1j*kuv3, 0, 2]
fbabisparamtab[6]=[ -kpeak3 + 1j*kuv3, 0, 2]
fbabisparamtab[7]=[ -kpeak4 - 1j*kuv4, 0, 1]
fbabisparamtab[8]=[ -kpeak4 + 1j*kuv4, 0, 1]
fbabisparamtab[9]=[ -kpeak1 - 1j*kuv1, 0, 1]
fbabisparamtab[10]=[ -kpeak1 + 1j*kuv1, 0, 1]
fbabisparamtab[11]=[ -kpeak2 - 1j*kuv2, 1, 2]
fbabisparamtab[12]=[ -kpeak2 + 1j*kuv2, 1, 2]
fbabisparamtab[13]=[ -kpeak3 - 1j*kuv3, 0, 3]
fbabisparamtab[14]=[ -kpeak3 + 1j*kuv3, 0, 3]
fbabisparamtab[15]=[ -kpeak4 - 1j*kuv4, 0, 2]
fbabisparamtab[16]=[ -kpeak4 + 1j*kuv4, 0, 2]
fbabisparamtab[17]=[ -kpeak1 - 1j*kuv1, 0, 2]
fbabisparamtab[18]=[ -kpeak1 + 1j*kuv1, 0, 2]
fbabisparamtab[19]=[ -kpeak2 - 1j*kuv2, 1, 3]
fbabisparamtab[20]=[ -kpeak2 + 1j*kuv2, 1, 3]
fbabisparamtab[21]=[ -kpeak3 - 1j*kuv3, 0, 4]
fbabisparamtab[22]=[ -kpeak3 + 1j*kuv3, 0, 4]
fbabisparamtab[23]=[ -kpeak4 - 1j*kuv4, 0, 3]
fbabisparamtab[24]=[ -kpeak4 + 1j*kuv4, 0, 3]
fbabisparamtab[25]=[ -kpeak1 - 1j*kuv1, 0, 3]
fbabisparamtab[26]=[ -kpeak1 + 1j*kuv1, 0, 3]
fbabisparamtab[27]=[ -kpeak2 - 1j*kuv2, 1, 4]
fbabisparamtab[28]=[ -kpeak2 + 1j*kuv2, 1, 4]
fbabisparamtab[29]=[ -kpeak3 - 1j*kuv3, 0, 5]
fbabisparamtab[30]=[ -kpeak3 + 1j*kuv3, 0, 5]
fbabisparamtab[31]=[ -kpeak4 - 1j*kuv4, 0, 4]
fbabisparamtab[32]=[ -kpeak4 + 1j*kuv4, 0, 4]

fbabisparamtab_masses_arr = fbabisparamtab[:,0].astype(mpc)
cdef mpc[:] fbabisparamtab_masses = fbabisparamtab_masses_arr

fbabisparamtab_exps_arr = fbabisparamtab[:,1:3].astype(int)
cdef long[:,:] fbabisparamtab_exps = fbabisparamtab_exps_arr

basis1tab_arr = np.array([9, 10, 17, 18, 25, 26], dtype = int)
cdef long[:] basis1tab = basis1tab_arr

basis2tab_arr = np.array([1, 2, 11, 12, 19, 20, 27, 28], dtype = int)
cdef long[:] basis2tab = basis2tab_arr

basis3tab_arr = np.array([3, 4, 5, 6, 13, 14, 21, 22, 29, 30], dtype = int)
cdef long[:] basis3tab = basis3tab_arr

basis4tab_arr = np.array([7, 8, 15, 16, 23, 24, 31, 32], dtype = int)
cdef long[:] basis4tab = basis4tab_arr

cdef long[:] giveIndices(int d):
	if d == 0:
		return np.array([0], dtype = int)
	if (d+1)%4 == 0:
		return basis4tab[0:int((d+1)/2)]
	if (d+1)%4 == 1:
		return basis1tab[0:int((d+4)/2)]
	if (d+1)%4 == 2:
		return basis2tab[0:int((d+3)/2)]
	if (d+1)%4 == 3:
		return basis3tab[0:int((d+2)/2)+2]

cdef list dtab = []
cdef Py_ssize_t i
for i in range(16):
	dtab.append(np.array(giveIndices(i), dtype = int))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef list build_matcoefs():
	cdef Py_ssize_t i 
	cdef list matcoefs = []
	cdef double complex[:] temp
	for i in range(16):
		temp = matdiogotobabisnowig[i]
		matcoefs.append(np.take(np.array(temp, dtype = complex), dtab[i]))
	return matcoefs

cdef list matcoefs = build_matcoefs()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double computeL(long[:] d1new, long[:] d2basis, long[:] d3basis, long n1, long n2, long n3, 
	long d1, long d2, long d3, mpfr k1sq, mpfr k2sq, mpfr k3sq):

	cdef long lend1 = len(d1new), 
	cdef long lend2 = len(d2basis)
	cdef long lend3 = len(d3basis)
	cdef double complex[:,:,:] matmul = np.empty((lend1, lend2, lend3), dtype = complex)

	cdef Py_ssize_t indx1, indx2, indx3, i, j, l
	coef_babis_d1_even_arr = np.array(matcoefs[d1][::2], dtype= complex)
	
	cdef double complex[:] coef_babis_d1_even = coef_babis_d1_even_arr
	cdef double complex coef_d1_indx1

	coef_babis_d2_arr =  np.array(matcoefs[d2], dtype= complex)
	cdef double complex[:] coef_babis_d2 = coef_babis_d2_arr
	cdef double complex coef_d2_indx2

	coef_babis_d3_arr =  np.array(matcoefs[d3], dtype= complex)
	cdef double complex[:] coef_babis_d3 = coef_babis_d3_arr
	cdef double complex coef_d3_indx3

	cdef long exp_num_i, exp_num_j, exp_num_l
	cdef long exp_den_i, exp_den_j, exp_den_l
	cdef mpc mass_i 
	cdef mpc mass_j 
	cdef mpc mass_l 

	cdef double complex Ltrain_temp
	
	cdef double complex add_result 
	cdef double complex result = 0. + 0.j
	# i, j and l are the babis functions indexes that make the Diogo function d1, d2 and d3
	for indx1 in range(lend1):
		i = d1new[indx1]
		coef_d1_indx1 = coef_babis_d1_even[indx1]
		exp_num_i = -n1 + fbabisparamtab_exps[i][0]
		exp_den_i =  fbabisparamtab_exps[i][1]
		mass_i = fbabisparamtab_masses[i]

		for indx2 in range(lend2):
			j = d2basis[indx2]
			coef_d2_indx2 = coef_babis_d2[indx2]
			exp_num_j = -n2 + fbabisparamtab_exps[j][0]
			exp_den_j =  fbabisparamtab_exps[j][1]
			mass_j = fbabisparamtab_masses[j]

			for indx3 in range(lend3):		
				l = d3basis[indx3]
				coef_d3_indx3 = coef_babis_d3[indx3]
				exp_num_l = - n3 + fbabisparamtab_exps[l][0]
				exp_den_l =  fbabisparamtab_exps[l][1]
				mass_l = fbabisparamtab_masses[l]

				#Ltrian_temp = <double complex>Ltrian(exp_num_j, exp_den_j, exp_num_i, exp_den_i, exp_num_l, exp_den_l, 
				#									k1sq, k2sq, k3sq, mass_j, mass_i, mass_l)
				matmul[indx1, indx2, indx3] = <double complex>Ltrian(-n2 + fbabisparamtab_exps[j][0], fbabisparamtab_exps[j][1], -n1 + 
					fbabisparamtab_exps[i][0], fbabisparamtab_exps[i][1], -n3 + fbabisparamtab_exps[l][0], fbabisparamtab_exps[l][1], k1sq, k2sq, k3sq, 
					fbabisparamtab_masses[j], fbabisparamtab_masses[i], fbabisparamtab_masses[l])
				#print(matmul[indx1,indx2, indx3])


				#add_result = coef_d1_indx1 * coef_d2_indx2 * coef_d3_indx3 * Ltrian_temp
				#result = result + add_result
	#return result.real/(4 * PI * SQRT_PI)	
	return 2*np.real(np.einsum('ijk, i, j, k', matmul, matcoefs[d1][::2], matcoefs[d2], matcoefs[d3]))/(8 * PI * SQRT_PI)	


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double computeL2(long[:] d2new, long[:] d3basis, long n1, long n2, long n3, 
		long d2, long d3, mpfr k1sq, mpfr k2sq, mpfr k3sq):

	cdef long lend2 = len(d2new), lend3 = len(d3basis)
	cdef double complex[:,:] matmul = np.empty((lend2, lend3), dtype = complex)
	cdef Py_ssize_t indx2, indx3, i, j

	for indx2 in range(lend2):
		for indx3 in range(lend3):
			i = d2new[indx2]
			j = d3basis[indx3]
			matmul[indx2, indx3] = <double complex>Ltrian(-n2 + fbabisparamtab_exps[i][0], fbabisparamtab_exps[i][1], -n1, 0, 
					-n3 + fbabisparamtab_exps[j][0], fbabisparamtab_exps[j][1], k1sq, k2sq, k3sq, 
					fbabisparamtab_masses[i], mpc(0), fbabisparamtab_masses[j])
			#print(matmul[indx2, indx3])

	return 2*np.real(np.einsum('ij, i, j', matmul, matcoefs[d2][::2], matcoefs[d3]))/(8 * PI * SQRT_PI)	

# cpdef double
def computeJ(long n1, long n2, long n3, 
					long d1, long d2, long d3, 
					mpfr k1sq, mpfr k2sq, mpfr k3sq):
	# n1, n2, n3 are the exponents of q, k1pq, k2mq in the denominator
	cdef double complex res = 0
	cdef long[:] d1basis = dtab[d1]
	cdef long[:] d2basis = dtab[d2]
	cdef long[:] d3basis = dtab[d3]
	cdef long[:] d3new = d3basis[::2]
	cdef long lend3 = len(d3new)
	cdef Py_ssize_t i, j, l, indx
	cdef double complex coef, term

	#print(matarray_val)

	if d1 == 0:
		if d2 == 0:
			if d3 == 0:
				return  float(Ltrian(-n2, 0, -n1, 0, -n3, 0, k1sq, k2sq, k3sq, 
					mpc(0), mpc(0), mpc(0)).real/(8  * PI * SQRT_PI))
			else:
				for indx in range(lend3):
					i = d3new[indx]
					coef = matdiogotobabisnowig[d3, i]
					term = <double complex>Ltrian(-n2, 0, -n1, 0, -n3 + 
					fbabisparamtab_exps[i][0], fbabisparamtab_exps[i][1], k1sq, k2sq, k3sq, 
					mpc(0), mpc(0), fbabisparamtab_masses[i])
					res += coef*term
				return float((2*res.real)/(8 * PI * SQRT_PI))			
		else:
			return computeL2(d2basis[::2], d3basis, n1, n2, n3, d2, d3, k1sq, k2sq, k3sq)
	else:
		return computeL(d1basis[::2], d2basis, d3basis, n1, n2, n3, d1, d2, d3, k1sq, k2sq, k3sq)