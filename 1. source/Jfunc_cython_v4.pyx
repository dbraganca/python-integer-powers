# cython: profile=True

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
import time

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

cdef double complex[:] matarray = np.fromfile('matdiogotobabis', np.complex128)
cdef double complex[:,:] matdiogotobabisnowig = np.reshape(matarray,(16, 33))

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

cdef mpc[:] fbabisparamtab_masses = fbabisparamtab[:,0].astype(mpc)
cdef long[:,:] fbabisparamtab_exps = fbabisparamtab[:,1:3].astype(int)

cdef long[:] basis1tab = np.array([9, 10, 17, 18, 25, 26], dtype = int)
cdef long[:] basis2tab = np.array([1, 2, 11, 12, 19, 20, 27, 28], dtype = int)
cdef long[:] basis3tab = np.array([3, 4, 5, 6, 13, 14, 21, 22, 29, 30], dtype = int)
cdef long[:] basis4tab = np.array([7, 8, 15, 16, 23, 24, 31, 32], dtype = int)

def giveIndices(d):
	if d == 0:
		return [0]
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

cdef list matcoefs = []
for i in range(16):
	temp = matdiogotobabisnowig[i]
	matcoefs.append(np.take(np.array(temp, dtype = complex), dtab[i]))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double computeL(long[:] d1new, long[:] d2basis, long[:] d3basis, long n1, long n2, long n3, 
	long d1, long d2, long d3, mpfr k1sq, mpfr k2sq, mpfr k3sq):

	cdef long lend1 = len(d1new), lend2 = len(d2basis), lend3 = len(d3basis)
	cdef double complex[:,:,:] matmul = np.empty((lend1, lend2, lend3), dtype = complex)
	cdef Py_ssize_t indx1, indx2, indx3, i, j, l

	for indx1 in range(lend1):
		for indx2 in range(lend2):
			for indx3 in range(lend3):
				i = d1new[indx1]
				j = d2basis[indx2]
				l = d3basis[indx3]
				matmul[indx1, indx2, indx3] = <double complex>Ltrian(-n2 + fbabisparamtab_exps[j][0], fbabisparamtab_exps[j][1], -n1 + 
					fbabisparamtab_exps[i][0], fbabisparamtab_exps[i][1], -n3 + fbabisparamtab_exps[l][0], fbabisparamtab_exps[l][1], k1sq, k2sq, k3sq, 
					fbabisparamtab_masses[j], fbabisparamtab_masses[i], fbabisparamtab_masses[l])
				#print(matmul[indx1,indx2, indx3])


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