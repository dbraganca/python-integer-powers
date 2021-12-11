
# without this - cython: profile=True
# maybe include this: distutils: language = c++

import cython
cimport cython

import numpy as np
cimport numpy as np

import gmpy2 as gm
from gmpy2 import atan, log, sqrt, gamma
from gmpy2 cimport *

from functools import lru_cache
from config import Ltrian_complex_cache, Ltrian_cache, TriaN_cache

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

import_gmpy2()   # needed to initialize the C-API


gm.get_context().precision = 190
gm.get_context().allow_complex = True
cdef mpfr_prec_t PREC = 190
cdef mpc_rnd_t MPCRND = MPC_RNDZZ
cdef mpfr_rnd_t MPFRND = MPFR_RNDZ

cdef mpfr CHOP_TOL =  GMPy_MPFR_New(PREC, NULL)
# cdef CHOP_TOL = mpfr(1.e-30)
#mpfr_init2(CHOP_TOL, PREC)
mpfr_set_d(MPFR(CHOP_TOL), 1e-30, MPFRND)

cdef mpfr PI =  GMPy_MPFR_New(PREC, NULL)
#mpfr_init2(PI, PREC)
mpfr_const_pi(MPFR(PI), MPFRND)

cdef mpfr SQRT_PI = sqrt(PI)
cdef mpc mpc0 = mpc(0)

# Utility function that checks if two complex numbers are almost equal, given the tolerance tol
@cython.boundscheck(False)
cdef int almosteq(mpc z1, mpc z2, mpfr tol):
	if abs(z1 - z2) < tol:
		return 1
	else:
		return 0

# Utility function to calculate binomial coefficient (we could also build a lookup table)
# Does not affect speed (so not worth it to build lookup table)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long long binomial_c(long long n, long long k):
	if k > n or k < 0:
		print("Warning: k bigger/negative than n in binomial_c")
		return 0
	if k == 0 or k == n:
		return 1

	if 2*k > n:
		return binomial_c(n, n - k)
	return n * binomial_c(n - 1, k - 1) / k

# Utility function to calculate trinomial coefficient
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long long trinomial(long long n, long long k, long long i):
	# coefficient in (x + y + z)^n corresponding to x^k y^i z^(n-k-i)
	return binomial_c(n, k + i) * binomial_c(k + i, i)

#The following 7 functions allow us to calculate dim reg results 
# when we have powers in the numerator

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long long get_coef_simple(long long n1, long long kmq2exp):
	# get coefficient of the expansion of ((k-q)^2+m)^n1 
	# that has the form ((k-q)^2)^kmq2exp * m^(n1-kmq2exp)
	# (aka just binomial expansion)
	return binomial_c(n1, kmq2exp)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef long long get_coef(long long n1, long long k2exp, long long q2exp, int kmq = True):
	# get coefficient of the expansion of (k-q)^2n1 
	# that has the form (k^2)^k2exp * (q^2)^q2exp * (k.q)^(n1-k2exp-q2exp)
	cdef long long kqexp = n1 - k2exp - q2exp
	cdef long long sign = 1
	if kmq:
		sign = (-1)**kqexp
	return sign * trinomial(n1, k2exp, q2exp) * 2**kqexp

@cython.boundscheck(False)
@cython.wraparound(False)
cdef num_terms(long long n1, int kmq = True):
	# expands terms of the type (k-q)^(2n1) if kmq = True
	# expands terms of the type (k+q)^(2n1) if kmq = False

	cdef long long list_length = (1 + n1)*(2 + n1)/2	
	cdef long long k2exp = 0, q2exp = 0
	cdef long long[:] term_list = np.zeros((list_length,), dtype = np.longlong)
	cdef long long[:,:] exp_list = np.zeros((list_length,3), dtype = np.longlong)

	cdef int i = 0
	for k2exp in range(n1+1):
		for q2exp in range(n1-k2exp+1):
			term_list[i] = get_coef(n1,k2exp,q2exp,kmq)
			exp_list[i,0] = k2exp
			exp_list[i,1] = q2exp
			exp_list[i,2] = n1-k2exp-q2exp
		#	print(term_list[i])
		#	print(exp_list[i,0],exp_list[i,1],exp_list[i,2])
			i += 1			
	return (term_list, exp_list)


coef_dim_gen_cache = {}
cdef long double coef_dim_gen(long long expnum, long long expden):
	# function to calculate coefficient in dim_gen without using Gamma functions 
	global coef_dim_gen_cache
	
	cdef long double lookup_val, val
		
	if expden == 1:
		return ((-1)**(expnum + 1))
	if expden < 1:
		return 0 
	if expden > 1:
		if (expnum,expden) in coef_dim_gen_cache:
			return coef_dim_gen_cache[(expnum,expden)]
		else:
			val = (coef_dim_gen(expnum, expden-1)*(5. - 2.*expden + 2.*expnum))/(2. - 2.*expden)
			coef_dim_gen_cache[(expnum,expden)] = val
			return val

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc dim_gen(long long expnum, long long expden, mpc m):
	if m == mpc0:
		return mpc0
	return (2.*SQRT_PI)*(coef_dim_gen(expnum,expden)*m**(expnum - expden + 1)*sqrt(m))

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef mpc dim_gen(long long expnum, long long expden, mpc m):
#	if m == mpc0:
#		return mpc0
#	return (2./SQRT_PI)*gamma(expnum+mpfr(1.5))*gamma(expden-expnum-mpfr(1.5))/gamma(expden)*m**(expnum - expden + 1)*sqrt(m)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef expand_massive_num(long long n1):
	#expand term of the form ((kmq)^2+mNum)^n1
	#returns:
	# - term_list: list of coefficients
	# - exp_list: list of exponents [expkmq2,mNumexp]

	cdef long list_length = n1 + 1
	cdef int i = 0
	cdef long kmq2exp = 0
	cdef long long[:] term_list = np.zeros((list_length,), dtype = np.longlong)
	cdef long long[:,:] exp_list = np.zeros((list_length,2), dtype = np.longlong)

	for kmq2exp in range(n1+1):
		term_list[i] = get_coef_simple(n1,kmq2exp)
		exp_list[i,0] = kmq2exp
		exp_list[i,1] = n1-kmq2exp
		i += 1

	return term_list, exp_list

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc dim_result(long expkmq2, long expden, mpfr k2, mpc mDen, int kmq = True):
	# computes integrals of the type ((k-q)^2)^expnum / (q^2 + mDen)^expden
	# if kmq is TRUE, consider (k-q)
	# if kmq is FALSE, consider (k+q)

	cdef int list_length = (1 + expkmq2)*(2 + expkmq2)/2
	cdef long long k2exp = 0, q2exp = 0, kqexp = 0
	cdef long long[:] term_list = np.zeros((list_length,), dtype = np.longlong)
	cdef long long[:,:] exp_list = np.zeros((list_length,3), dtype = np.longlong)
	
	term_list, exp_list = num_terms(expkmq2,kmq)
	
	cdef mpc res_list = mpc0

	cdef Py_ssize_t i = 0
	for i in range(list_length):
		k2exp = exp_list[i][0]
		q2exp = exp_list[i][1]
		kqexp = exp_list[i][2]

		if kqexp%2 == 0:
			if kqexp != 0:
				res_list += term_list[i]*dim_gen(q2exp+kqexp/2,expden,mDen)*(k2)**(k2exp+kqexp/2)/(1+kqexp)
			
			else:
				res_list += term_list[i]*dim_gen(q2exp,expden,mDen)*(k2)**(k2exp)
	return res_list

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc compute_massive_num(long long expnum, long long expden, 
							mpfr k2, mpc mNum, mpc mDen, 
							int kmq = True):
	# computes integrals of the type ((k-q)^2+mNum)^expnum / (q^2 + mDen)^expden
	# by summing over many dim_results
	
	if mNum == 0:
		return dim_result(expnum,expden,k2,mDen)


	cdef Py_ssize_t list_length = expnum + 1
	cdef long long kmq2exp = 0, mNumexp = 0
	cdef long long[:] term_list = np.zeros((list_length,), dtype = np.longlong)
	cdef long long[:,:] exp_list = np.zeros((list_length,2), dtype = np.longlong)
	term_list, exp_list = expand_massive_num(expnum)
	
	cdef mpc res_list = mpc0

	cdef Py_ssize_t i
	for i in range(list_length):
		kmq2exp = exp_list[i][0]
		mNumexp = exp_list[i][1]

		res_list += term_list[i]*(mNum**mNumexp)*dim_result(kmq2exp,expden,k2,mDen)

	return res_list


#Function that calculates Tadpoles
@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc TadN(long n, mpc m):
	if m == mpc0:
		return mpc0
	if n == 0:
		return mpc0
	if n == 1:
		return -2 * (SQRT_PI * sqrt(m))
	if n < 0:
		return mpc0
	return dim_gen(0,n,m)

@cython.boundscheck(False)
@cython.wraparound(False)
@lru_cache(None)
def BubN(long n1, long n2, mpfr k2, mpc m1, mpc m2):

	if n1 == 0:
		# print('n2', n2, 'm2', m2, 'TadN', TadN(n2, m2))
		return TadN(n2, m2)

	if n2 == 0:
		# print('n1', n1, 'm1', m1, 'TadN',TadN(n1, m1))
		return TadN(n1, m1)

	if n1 == 1 and n2 == 1:
		# print('BubMaster',BubMaster(k2, m1, m2))
		return BubMaster(k2, m1, m2)

	cdef mpc k1s = k2 + m1 + m2
	cdef mpc jac = k1s**2 - 4*m1*m2
	cdef mpc cpm0, cmp0, c000
	cdef long dim = 3
	cdef long long nu1, nu2, Ndim

	if n1 > 1:
		nu1 = n1 - 1
		nu2 = n2
		Ndim = dim - nu1 - nu2

		cpm0 = k1s
		cmp0 = -2.*m2/nu1*nu2
		c000 = (2.*m2-k1s)/nu1*Ndim - 2.*m2 + (k1s*nu2)/nu1
		#c000 = 2*m2*(1./nu1 * Ndim - 1) + k1s * (1./nu1 * (nu2 - Ndim))
	elif n2 > 1:
		nu1 = n1
		nu2 = n2-1
		Ndim = dim - nu1 - nu2

		cpm0 = -2*m1/nu2*nu1
		cmp0 = k1s
		c000 = (2.*m1 - k1s)/nu2*Ndim + (k1s*nu1)/nu2 - 2.*m1
		#c000 = 2*m1*(1./nu2 * Ndim - 1) + k1s * (1./nu2 * (nu1 - Ndim)) 
	#code to deal with numerators
	if n1 < 0 or n2 < 0:
		if m1 == mpc0 and m2 == mpc0:
			return mpc0
		if n1 < 0 and n2 > 0:
			# m1 is the mass in the numerator
			# m2 is the mass in the denominator 
			return compute_massive_num(-n1,n2,k2,m1,m2)
		elif n2 < 0 and n1 > 0:
			# m2 is the mass in the numerator
			# m1 is the mass in the denominator
			return compute_massive_num(-n2,n1,k2,m2,m1)
		else:
			# case of NO DENOMINATOR
			return 0		

	c000 = c000/jac
	cmp0 = cmp0/jac
	cpm0 = cpm0/jac

	if c000 == 0.:
		return cpm0*BubN(nu1 + 1, nu2 - 1, k2, m1, m2) + cmp0*BubN(nu1 - 1, nu2 + 1, k2, m1, m2)
	
	return c000*BubN(nu1,nu2,k2,m1,m2) + cpm0*BubN(nu1 + 1, nu2 - 1, k2, m1, m2) + cmp0*BubN(nu1 - 1, nu2 + 1, k2, m1, m2)
 
@cython.boundscheck(False)
@cython.wraparound(False)
@lru_cache(None)
def TrianKinem(mpfr k21, mpfr k22, mpfr k23, 
				mpc m1, mpc m2, mpc m3):
	
	cdef mpc k1s, k2s, k3s, jac, ks11, ks12, ks22, ks23, ks31, ks33
	
	k1s = k21 + m1 + m2
	k2s = k22 + m2 + m3
	k3s = k23 + m3 + m1

	jac = -4*m1*m2*m3 + k1s**2*m3 + k2s**2*m1 + k3s**2*m2 - k1s*k2s*k3s
	jac = 2*jac

	ks11 = (-4*m1*m2 + k1s**2)/jac
	ks12 = (-2*k3s*m2 + k1s*k2s)/jac
	ks22 = (-4*m2*m3 + k2s**2)/jac
	ks23 = (-2*k1s*m3 + k2s*k3s)/jac
	ks31 = (-2*k2s*m1+k1s*k3s)/jac
	ks33 = (-4*m1*m3+k3s**2)/jac

	#cdef mpc[:] kinems = np.array([jac,ks11,ks22,ks33,ks12,ks23,ks31], dtype = mpc)
	#kinems = [jac,ks11,ks22,ks33,ks12,ks23,ks31]
	return  np.array([jac,ks11,ks22,ks33,ks12,ks23,ks31], dtype = mpc)

# this functions AFFECTS SPEED: decreases speed by 1/3

@cython.boundscheck(False)
@cython.wraparound(False)
#@lru_cache(None)
cdef mpc TriaN(long n1, long n2, long n3, 
			mpfr k21, mpfr k22, mpfr k23, 
			mpc m1, mpc m2, mpc m3, 
			long m1_ind, long m2_ind, long m3_ind):
	# print("n1, n2, n3", n1, n2, n3)
	# print(n1,d1,n2,d2,n3,d3,m1,m2,m3)

	arg_list = (n1,n2,n3,m1_ind,m2_ind,m3_ind)
	# check cache
	if arg_list in TriaN_cache:
		return TriaN_cache[arg_list]

	# reduce to Bubble integral
	if n1 == 0:
		return BubN(n2, n3, k22, m2, m3)
	if n2 == 0:
		return BubN(n3, n1, k23, m3, m1)
	if n3 == 0:
		return BubN(n1, n2, k21, m1, m2)

	if n1 == 1 and n2 == 1 and n3 == 1:
		return TriaMaster(k21, k22, k23, m1, m2, m3)

	if n1 < 0 or n2 < 0 or n3 < 0:
		if n1 < -4 or n2 < -4 or n3 < -4:
			print('ERROR: case not considered -  n1, n2, n3', n1,n2,n3)
		if n1 < 0:
			if n2 > 0 and n3 > 0:
				result = tri_dim(-n1,n2,n3,k21,k22,k23,m2,m3)
				TriaN_cache[arg_list] = result
				return result
			elif n2 < 0: 
				result = tri_dim_two(-n2,-n1,n3,k22,k23,k21,m3)
				TriaN_cache[arg_list] = result
				return result
			else:
				result = tri_dim_two(-n1,-n3,n2,k21,k22,k23,m2)
				TriaN_cache[arg_list] = result
				return result
		if n2 < 0:
			if n1 > 0 and n3 > 0:
				result = tri_dim(-n2,n1,n3,k21,k23,k22,m1,m3)
				TriaN_cache[arg_list] = result
				return result
			if n3 < 0:
				result =  tri_dim_two(-n3,-n2,n1,k23,k21,k22,m1)
				TriaN_cache[arg_list] = result
				return result
		if n3 < 0:
			if n1 > 0 and n2 > 0:
				result = tri_dim(-n3,n1,n2,k23,k21,k22,m1,m2)	
				TriaN_cache[arg_list] = result
				return result
			print('ERROR: case not considered')

	cdef long long nu1, nu2, nu3, Ndim, dim = 3
	cdef mpc ks11, ks22, ks33, ks12, ks23, ks31
	cdef mpc cpm0, cmp0, cm0p, cp0m, c0pm, c0mp, c000

	cdef mpc[:] kinem = TrianKinem(k21, k22, k23, m1, m2, m3)
	#jac = kinem[0]
	ks11 = kinem[1]
	ks22 = kinem[2]
	ks33 = kinem[3]
	ks12 = kinem[4]
	ks23 = kinem[5]
	ks31 = kinem[6]

	if n1 > 1:
		nu1 = n1 - 1
		nu2 = n2
		nu3 = n3

		Ndim = dim - nu1 - nu2 - nu3

		cpm0 = -ks23
		cmp0 = (ks22*nu2)/nu1
		cm0p = (ks22*nu3)/nu1
		cp0m = -ks12
		c0pm = -(ks12*nu2)/nu1
		c0mp = -(ks23*nu3)/nu1
		c000 = (-nu3+Ndim)*ks12/nu1 - (-nu1+Ndim)*ks22/nu1 + (-nu2+Ndim)*ks23/nu1

	elif n2 > 1:
		nu1 = n1
		nu2 = n2 - 1 
		nu3 = n3

		Ndim = dim - nu1 - nu2 - nu3

		cpm0 = (ks33*nu1)/nu2
		cmp0 = -ks23
		cm0p = -(ks23*nu3)/nu2
		cp0m = -(ks31*nu1)/nu2
		c0pm = -ks31
		c0mp = (ks33*nu3)/nu2
		c000 = (-nu1 + Ndim)*ks23/nu2 + (-nu3 + Ndim)*ks31/nu2 - (-nu2 + Ndim)*ks33/nu2

	elif n3 > 1:
		nu1 = n1
		nu2 = n2
		nu3 = n3 - 1 

		Ndim = dim - nu1 - nu2 - nu3


		cpm0 = -(ks31*nu1)/nu3
		cmp0 = -(ks12*nu2)/nu3
		cm0p = -ks12
		cp0m = (ks11*nu1)/nu3
		c0pm = (ks11*nu2)/nu3
		c0mp = -ks31
		c000 = -(-nu3 + Ndim)*ks11/nu3 + (-nu1 + Ndim)*ks12/nu3 + (-nu2 + Ndim)*ks31/nu3
	
	result = (c000*TriaN(nu1, nu2, nu3, k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind) 
			+ c0mp*TriaN(nu1, nu2-1, nu3+1, k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind) 
			+ c0pm*TriaN(nu1, nu2+1, nu3-1, k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind)
			+ cm0p*TriaN(nu1-1, nu2, nu3+1, k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind)
			+ cp0m*TriaN(nu1+1, nu2, nu3-1, k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind) 
			+ cmp0*TriaN(nu1-1, nu2+1, nu3, k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind)
			+ cpm0*TriaN(nu1+1, nu2-1, nu3, k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind))

	TriaN_cache[arg_list] = result
	return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc tri_dim(long long n1, long long d1, long long d2, 
				mpfr numk2, mpfr denk2, mpfr ksum2, 
				mpc m1, mpc m2):
	# integral of (numk-q)^2n1/(q^2+m1)^2d1/((denk+q)^2+m2)^2d2
	#numerator (numk-q)^2n1 is massless
	#m1 is mass of d1 propagator, which is (q^2+m1)^2d1,
	#m2 is mass of d2 propagator, which is ((denk+q)^2+m2)^2d2

	
	cdef Py_ssize_t list_length = (1 + n1)*(2 + n1)/2	
	cdef long long k2exp = 0, q2exp = 0, kqexp = 0
	cdef long long[:] term_list = np.zeros((list_length,), dtype = np.longlong)
	cdef long long[:,:] exp_list = np.zeros((list_length,3), dtype = np.longlong)
	term_list, exp_list = num_terms(n1,True)

	cdef mpc res_list = mpc0
	cdef mpc term = mpc0
	
	cdef Py_ssize_t i 
	for i in range(list_length):
		k2exp = exp_list[i][0]
		q2exp = exp_list[i][1]
		kqexp = exp_list[i][2]
		if kqexp == 0:
			# in this case our numerator is just (q2)^q2exp
			if q2exp == 0:
				term = BubN(d1,d2,denk2,m1,m2)
			elif q2exp == 1:
				term = BubN(d1-1,d2,denk2,m1,m2)-m1*BubN(d1,d2,denk2,m1,m2)
			elif q2exp == 2:
				term = BubN(d1-2,d2,denk2,m1,m2) -2*m1*BubN(d1-1,d2,denk2,m1,m2) + m1**2*BubN(d1,d2,denk2,m1,m2)
			elif q2exp == 3:
				term = BubN(d1-3,d2,denk2,m1,m2) - 3*m1*BubN(d1-2,d2,denk2,m1,m2) + 3*m1**2*BubN(d1-1,d2,denk2,m1,m2) - m1**3*BubN(d1,d2,denk2,m1,m2)
			elif q2exp == 4:
				term = BubN(d1-4,d2,denk2,m1,m2) - 4*m1*BubN(d1-3,d2,denk2,m1,m2) + 6*m1**2*BubN(d1-2,d2,denk2,m1,m2) - 4*m1**3*BubN(d1-1,d2,denk2,m1,m2) + m1**4*BubN(d1,d2,denk2,m1,m2)
			else:
				print('exceeded calculable power')

		elif kqexp == 1:
			if q2exp == 0:
				term = num_one_pow(d1,d2,denk2,m1,m2)*k1dotk2(numk2,denk2,ksum2)
			elif q2exp == 1:
				term = (num_one_pow(d1-1,d2,denk2,m1,m2)-m1*num_one_pow(d1,d2,denk2,m1,m2))*k1dotk2(numk2,denk2,ksum2)
			elif q2exp == 2:
				term = (num_one_pow(d1-2,d2,denk2,m1,m2) - 2*m1*num_one_pow(d1-1,d2,denk2,m1,m2) + m1**2*num_one_pow(d1,d2,denk2,m1,m2))*k1dotk2(numk2,denk2,ksum2)
			elif q2exp == 3:
				term = (num_one_pow(d1-3,d2,denk2,m1,m2) - 3*m1*num_one_pow(d1-2,d2,denk2,m1,m2) + 3*m1**2*num_one_pow(d1-1,d2,denk2,m1,m2) - m1**3*num_one_pow(d1,d2,denk2,m1,m2))*k1dotk2(numk2,denk2,ksum2)
			else:
				print('exceeded calculable power')
							
			# print('term after second if', term)
		elif kqexp == 2:
			delta_coef, dkcoef = num_two_pow(d1,d2,denk2,m1,m2)
			if q2exp == 0:
				term = (numk2*delta_coef + k1dotk2(numk2,denk2,ksum2)**2/denk2*dkcoef)
			elif q2exp == 1:
				delta_coef2, dkcoef2 = num_two_pow(d1-1,d2,denk2,m1,m2)
				term = -m1*(numk2*delta_coef + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef)
				term += (numk2*delta_coef2 + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef2)
			elif q2exp == 2:
				delta_coef2, dkcoef2 = num_two_pow(d1-1,d2,denk2,m1,m2)
				delta_coef3, dkcoef3 = num_two_pow(d1-2,d2,denk2,m1,m2)
				term = (numk2*delta_coef3 + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef3)
				term += -2*m1*(numk2*delta_coef2 + k1dotk2(numk2,denk2,ksum2)**2/(denk2)*dkcoef2)
				term += m1**2*(numk2*delta_coef + k1dotk2(numk2,denk2,ksum2)**2/denk2*dkcoef)
			else:
				print('exceeded calculable power')
		
		elif kqexp == 3:
			delta_coef, dkcoef = num_three_pow(d1,d2,denk2,m1,m2)
			if q2exp == 0:				
				term = (numk2*delta_coef*k1dotk2(numk2,denk2,ksum2)/(sqrt(denk2)) + dkcoef*k1dotk2(numk2,denk2,ksum2)**3/(denk2*sqrt(denk2)))
			elif q2exp == 1:
				delta_coef2, dkcoef2 = num_three_pow(d1-1,d2,denk2,m1,m2)
				term = (numk2*delta_coef2*k1dotk2(numk2,denk2,ksum2)/(sqrt(denk2)) + dkcoef2*k1dotk2(numk2,denk2,ksum2)**3/(denk2*sqrt(denk2)))
				term += -m1*(numk2*delta_coef*k1dotk2(numk2,denk2,ksum2)/(sqrt(denk2)) + dkcoef*k1dotk2(numk2,denk2,ksum2)**3/(denk2*sqrt(denk2)))
			else:
				print('exceeded calculable power')

		elif kqexp == 4:
			# print('using power 4')	
			if q2exp == 0:
				coef1, coef2, coef3 = num_four_pow(d1,d2,denk2,m1,m2)
				term = coef1*numk2**2 + numk2*k1dotk2(numk2,denk2,ksum2)**2*coef2/denk2 + coef3*k1dotk2(numk2,denk2,ksum2)**4/denk2**2
			else:
				print(kqexp, q2exp, 'kqexp, q2exp')
				print('exceeded calculable power')

		if kqexp > 4:
			print(kqexp, q2exp, 'kqexp, q2exp')
			print('exceeded calculable power')
		
		res_list += term*term_list[i]*numk2**(k2exp)
	return res_list

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc num_one_pow(long long d1, long long d2, mpfr denk2, mpc m1, mpc m2):
	#integral of k.q/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by sqrt(k^2)
	# coef in front of k_i
	cdef mpc coef = BubN(d1,d2-1,denk2,m1,m2) - BubN(d1-1,d2,denk2,m1,m2) - (denk2 + m2 - m1)*BubN(d1,d2,denk2,m1,m2)
	coef = coef/(2*denk2)
	return coef

@cython.boundscheck(False)
@cython.wraparound(False)
cdef num_two_pow(long long d1, long long d2, mpfr denk2, mpc m1, mpc m2):
	#integral of (k.q)^2/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by k^2
	#denk2 are magnitudes of external momenta
	cdef mpc coef1 = -(BubN(d1,d2-2,denk2,m1,m2) - 2*(denk2 + m2 - m1)*BubN(d1,d2-1,denk2,m1,m2) + (denk2 + m2 - m1)**2*BubN(d1,d2,denk2,m1,m2)
		-2*BubN(d1-1,d2-1,denk2,m1,m2) + 2*(denk2 + m2 - m1)*BubN(d1-1,d2,denk2,m1,m2) + BubN(d1-2,d2,denk2,m1,m2))/(8*denk2) + BubN(d1-1,d2,denk2,m1,m2)/2 - m1*BubN(d1,d2,denk2,m1,m2)/2

	cdef mpc coef2 = BubN(d1-1,d2,denk2,m1,m2) - m1*BubN(d1,d2,denk2,m1,m2) - 3*coef1
	return coef1, coef2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef num_three_pow(long long d1, long long d2, mpfr denk2, mpc m1, mpc m2):
	#integral of (k.q)^3/(((q^2+m1)^d1)*((denk+q)^2+m2)^d2) divided by k^3

	cdef mpc aux0=((3*(m1-m2)*(m1-m2)+(2.*(denk2*(m1+m2))))-(denk2**2))*(BubN(-1 + d1, d2, denk2, m1, m2))
	cdef mpc aux1=((denk2**2)+((((m1-m2)*(m1-m2)))+(2.*(denk2*(m1+m2)))))*(BubN(d1, d2, denk2, m1, m2))
	cdef mpc aux2=(3*(((denk2+m2)-m1)*(BubN(d1, -2 + d2, denk2, m1, m2))))+(((denk2+m2)-m1)*aux1)
	cdef mpc aux3=(-2.*((denk2+((-3*m1)+(3*m2)))*(BubN(-1 + d1, -1 + d2, denk2, m1, m2))))+(aux0+aux2)
	cdef mpc aux4=(-3*(BubN(-2 + d1, -1 + d2, denk2, m1, m2)))+((3*(BubN(-1 + d1, -2 + d2, denk2, m1, m2)))+aux3)
	cdef mpc aux5=((3*(denk2**2))+((-2.*(denk2*(m1+(-3*m2))))+(3*(((m1-m2)*(m1-m2))))))*(BubN(d1, -1 + d2, denk2, m1, m2))
	cdef mpc aux6=((((BubN(-3 + d1, d2, denk2, m1, m2))+aux4)-aux5)-(BubN(d1, -3 + d2, denk2, m1, m2)))-((denk2+((3*m1)+(-3*m2)))*(BubN(-2 + d1, d2, denk2, m1, m2)))
	cdef mpc coef1=3*aux6/(16 * denk2 * sqrt(denk2))

	cdef mpc coef2 = 1/(2*sqrt(denk2))*(BubN(d1-1,d2-1,denk2,m1,m2) - BubN(d1-2,d2,denk2,m1,m2)
		-(denk2 + m2 - 2*m1)*BubN(d1-1,d2,denk2,m1,m2) - m1*BubN(d1,d2-1,denk2,m1,m2)
		+(denk2 + m2 - m1)*m1*BubN(d1,d2,denk2,m1,m2))-5*coef1/3
	return coef1, coef2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef num_four_pow(long long d1, long long d2, 
					mpfr denk2, mpc m1, mpc m2):

	cdef mpc aux0=((3*(((denk2+m1)**2)))+((-2.*((denk2+(3*m1))*m2))+(3*(m2**2))))*(BubN(-2 + d1, d2, denk2, m1, m2))
	cdef mpc aux1=((denk2**2)+((-3*(((m1-m2)**2)))+(-2.*(denk2*(m1+m2)))))*(BubN(-1 + d1, -1 + d2, denk2, m1, m2))
	cdef mpc aux2=((denk2**2)+((((m1-m2)**2))+(2.*(denk2*(m1+m2)))))*(BubN(-1 + d1, d2, denk2, m1, m2))
	cdef mpc aux3=((3*(denk2**2))+((-2.*(denk2*(m1+(-3*m2))))+(3*(((m1-m2)**2)))))*(BubN(d1, -2 + d2, denk2, m1, m2))
	cdef mpc aux4=((denk2**2)+((((m1-m2)**2))+(2.*(denk2*(m1+m2)))))*(BubN(d1, -1 + d2, denk2, m1, m2))
	cdef mpc aux5=((((denk2**2)+((((m1-m2)**2))+(2.*(denk2*(m1+m2)))))**2))*(BubN(d1, d2, denk2, m1, m2))
	cdef mpc aux6=(-4.*(((denk2+m2)-m1)*(BubN(d1, -3 + d2, denk2, m1, m2))))+((2.*aux3)+((-4.*(((denk2+m2)-m1)*aux4))+aux5))
	cdef mpc aux7=(4.*aux1)+((-4.*(((denk2+m1)-m2)*aux2))+((BubN(d1, -4 + d2, denk2, m1, m2))+aux6))
	cdef mpc aux8=(4.*((denk2+((-3*m1)+(3*m2)))*(BubN(-1 + d1, -2 + d2, denk2, m1, m2))))+aux7
	cdef mpc aux9=(4.*((denk2+((3*m1)+(-3*m2)))*(BubN(-2 + d1, -1 + d2, denk2, m1, m2))))+((2.*aux0)+((-4.*(BubN(-1 + d1, -3 + d2, denk2, m1, m2)))+aux8))
	cdef mpc aux10=(-4.*(((denk2+m1)-m2)*(BubN(-3 + d1, d2, denk2, m1, m2))))+((6.*(BubN(-2 + d1, -2 + d2, denk2, m1, m2)))+aux9)
	cdef mpc aux11=(BubN(-4 + d1, d2, denk2, m1, m2))+((-4.*(BubN(-3 + d1, -1 + d2, denk2, m1, m2)))+aux10)
	cdef mpc coef1=3*aux11/(128*denk2*denk2)

	aux0=((denk2**2)+((-15.*(((m1-m2)**2)))+(-6.*(denk2*(m1+m2)))))*(BubN(-2 + d1, d2, denk2, m1, m2))
	aux1=-12.*(((3*denk2)+((-5.*m1)+(5.*m2)))*(BubN(-1 + d1, -2 + d2, denk2, m1, m2)))
	aux2=((denk2**2)+((-2.*(denk2*(m1+(-3*m2))))+(5.*(((m1-m2)**2)))))*(BubN(-1 + d1, -1 + d2, denk2, m1, m2))
	aux3=((denk2**3)+((5.*((m1-m2)**3))+(3*(denk2*((m1-m2)*(m1+(3*m2)))))))-((denk2**2)*(m1+(3*m2)))
	aux4=((5.*(denk2**2))+((5.*(((m1-m2)**2)))+(denk2*((-6.*m1)+(10.*m2)))))*(BubN(d1, -2 + d2, denk2, m1, m2))
	aux5=((5.*(denk2**2))+((5.*(((m1-m2)**2)))+(2.*(denk2*(m1+(5.*m2))))))*(BubN(d1, -1 + d2, denk2, m1, m2))
	aux6=(20.*(((denk2+m2)-m1)*(BubN(d1, -3 + d2, denk2, m1, m2))))+((-6.*aux4)+(4.*(((denk2+m2)-m1)*aux5)))
	aux7=(4.*(aux3*(BubN(-1 + d1, d2, denk2, m1, m2))))+((-5.*(BubN(d1, -4 + d2, denk2, m1, m2)))+aux6)
	aux8=(2.*aux0)+((20.*(BubN(-1 + d1, -3 + d2, denk2, m1, m2)))+(aux1+((12.*aux2)+aux7)))
	aux9=(12.*((denk2+((-5.*m1)+(5.*m2)))*(BubN(-2 + d1, -1 + d2, denk2, m1, m2))))+aux8
	aux10=(4.*((denk2+((5.*m1)+(-5.*m2)))*(BubN(-3 + d1, d2, denk2, m1, m2))))+((-30.*(BubN(-2 + d1, -2 + d2, denk2, m1, m2)))+aux9)
	aux11=(-5.*(BubN(-4 + d1, d2, denk2, m1, m2)))+((20.*(BubN(-3 + d1, -1 + d2, denk2, m1, m2)))+aux10)
	cdef mpc aux12=((5.*(denk2**2))+((5.*(((m1-m2)**2)))+(denk2*((-6.*m1)+(10.*m2)))))*(BubN(d1, d2, denk2, m1, m2))
	cdef mpc aux13=(denk2**-2.)*(aux11-(((denk2**2)+((((m1-m2)**2))+(2.*(denk2*(m1+m2)))))*aux12))
	cdef mpc coef2=3*aux13/64

	aux0=-60.*(((3*denk2)+((-7.*m1)+(7.*m2)))*(BubN(-2 + d1, -1 + d2, denk2, m1, m2)))
	aux1=((3*(denk2**2))+((-10.*(denk2*(m1+(-3*m2))))+(35.*(((m1-m2)**2)))))*(BubN(-2 + d1, d2, denk2, m1, m2))
	aux2=((3*(denk2**2))+((7.*(((m1-m2)**2)))+(denk2*((-6.*m1)+(10.*m2)))))*(BubN(-1 + d1, -1 + d2, denk2, m1, m2))
	aux3=(-9.*((denk2**2)*(m1+(-5.*m2))))+((15.*(denk2*((m1+(-5.*m2))*(m1-m2))))+(-35.*((m1-m2)**3)))
	aux4=((7.*(denk2**2))+((7.*(((m1-m2)**2)))+(2.*(denk2*((-5.*m1)+(7.*m2))))))*(BubN(d1, -2 + d2, denk2, m1, m2))
	aux5=((7.*(denk2**2))+((-2.*(denk2*(m1+(-7.*m2))))+(7.*(((m1-m2)**2)))))*(((denk2+m2)-m1)*(BubN(d1, -1 + d2, denk2, m1, m2)))
	aux6=(35.*((m1-m2)**4.))+(6.*((denk2**2)*((3*(m1**2))+((-30.*(m1*m2))+(35.*(m2**2))))))
	aux7=(-20.*((denk2**3)*(m1+(-7.*m2))))+((-20.*(denk2*((m1+(-7.*m2))*(((m1-m2)**2)))))+aux6)
	aux8=(30.*aux4)+((-20.*aux5)+(((35.*(denk2**4.))+aux7)*(BubN(d1, d2, denk2, m1, m2))))
	aux9=(35.*(BubN(d1, -4 + d2, denk2, m1, m2)))+((-140.*(((denk2+m2)-m1)*(BubN(d1, -3 + d2, denk2, m1, m2))))+aux8)
	aux10=(-60.*aux2)+((4.*(((5.*(denk2**3))+aux3)*(BubN(-1 + d1, d2, denk2, m1, m2))))+aux9)
	aux11=(60.*(((5.*denk2)+((-7.*m1)+(7.*m2)))*(BubN(-1 + d1, -2 + d2, denk2, m1, m2))))+aux10
	aux12=(210.*(BubN(-2 + d1, -2 + d2, denk2, m1, m2)))+(aux0+((6.*aux1)+((-140.*(BubN(-1 + d1, -3 + d2, denk2, m1, m2)))+aux11)))
	aux13=(-140.*(BubN(-3 + d1, -1 + d2, denk2, m1, m2)))+((20.*((denk2+((-7.*m1)+(7.*m2)))*(BubN(-3 + d1, d2, denk2, m1, m2))))+aux12)
	cdef mpc coef3=((35.*(BubN(-4 + d1, d2, denk2, m1, m2)))+aux13)/(128*denk2*denk2)

	return coef1, coef2, coef3




#@lru_cache(None)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc tri_dim_two(long n1, long n2, long d1, 
				mpfr numk21, mpfr numk22, mpfr ksum2, 
				mpc dm):
	# integral of (k1 - q)^2^n1 (k2 + q)^2^n2/(q2+dm)^d1

	# term_list1 are the coefficients of (k1 - q)^2^n1 corresponding to the exponents in exp_list   
	# exp_list1 are the exponents of (k1 - q)^2^n1 of the form k1^2^k2exp1*q^2^q2exp1*(k.q)^kqexp1, 
	# written as (k2exp1, q2exp1, kqexp1) 

	# term_list2 are the coefficients of (k2 + q)^2^n2 corresponding to the exponents in exp_list   
	# exp_list2 are the exponents of (k2 + q)^2^n2 of the form k2^2^k2exp2*q^2^q2exp2*(k.q)^kqexp2, 
	# written as (k2exp2, q2exp2, kqexp2) 

	cdef Py_ssize_t list_length_n1 = (1 + n1)*(2 + n1)/2
	cdef Py_ssize_t list_length_n2 = (1 + n2)*(2 + n2)/2	
	cdef long k2exp1 = 0, q2exp1 = 0, k2exp2 = 0, q2exp2 = 0
	cdef long k2exp, q2exp, kqexp1, kqexp2, kqexp
	
	term_list1_arr = np.zeros((list_length_n1,), dtype = np.longlong)
	cdef long long[:] term_list1 = term_list1_arr
		
	exp_list1_arr = np.zeros((list_length_n1,3), dtype = np.longlong)
	cdef long long[:,:] exp_list1 = exp_list1_arr

	cdef long long[:] term_list2 = np.zeros((list_length_n2,), dtype = np.longlong)
	cdef long long[:,:] exp_list2 = np.zeros((list_length_n2,3), dtype = np.longlong)

	term_list1, exp_list1 = num_terms(n1,True)
	term_list2, exp_list2 = num_terms(n2,False)

	cdef mpc res_list = mpc0
	cdef mpc term = mpc0

	cdef Py_ssize_t i1, i2
	for i1 in range(list_length_n1):
		for i2 in range(list_length_n2):
			k2exp1 = exp_list1[i1][0]
			k2exp2 = exp_list2[i2][0]
			q2exp = exp_list1[i1][1] + exp_list2[i2][1]
			kqexp1 = exp_list1[i1][2]
			kqexp2 = exp_list2[i2][2]
			kqexp = kqexp1 + kqexp2

			# term = 0
			if kqexp%2 == 0:
				# if kqexp is odd then the integral vanishes by symmetry q -> -q				
				# if kqexp == 8:
				#	print('using power 8')				
				if kqexp != 0:
					#cases where kqexp == 2
					if kqexp1 == 2 and kqexp2 == 0:
						term = dim_gen(q2exp+1,d1,dm)*(numk21)**(kqexp1/2)/3
					elif kqexp1 == 0 and kqexp2 == 2:
						term = dim_gen(q2exp+1,d1,dm)*(numk22)**(kqexp2/2)/3
					elif kqexp1 == 1 and kqexp2 == 1:
						term = dim_gen(q2exp+1,d1,dm)*(k1dotk2(numk21,numk22,ksum2))/3

					# cases where kqexp == 4
					elif kqexp1 == 0 and kqexp2 == 4:
						term = dim_gen(q2exp+2,d1,dm)*(numk22**2)/5
					elif kqexp1 == 4 and kqexp2 == 0:
						term = dim_gen(q2exp+2,d1,dm)*(numk21**2)/5
					elif kqexp1 == 1 and kqexp2 == 3:
						term = dim_gen(q2exp+2,d1,dm)*(k1dotk2(numk21,numk22,ksum2)*numk22)/5
					elif kqexp1 == 3 and kqexp2 == 1:
						term = dim_gen(q2exp+2,d1,dm)*(k1dotk2(numk21,numk22,ksum2)*numk21)/5
					elif kqexp1 == 2 and kqexp2 == 2:
						term = dim_gen(q2exp+2,d1,dm)*(numk21*numk22 + 2*(k1dotk2(numk21,numk22,ksum2))**2)/15


					# cases where kqexp == 6
					elif kqexp1 == 6 and kqexp2 == 0:
						term = dim_gen(q2exp + 3, d1, dm)*numk21**3/7
					elif kqexp1 == 0 and kqexp2 == 6:
						term = dim_gen(q2exp + 3, d1, dm)*numk22**3/7
					elif kqexp1 == 5 and kqexp2 == 1:
						term = dim_gen(q2exp + 3, d1, dm)*numk21**2*k1dotk2(numk21,numk22,ksum2)/7
					elif kqexp1 == 1 and kqexp2 == 5:
						term = dim_gen(q2exp + 3, d1, dm)*numk22**2*k1dotk2(numk21,numk22,ksum2)/7
					elif kqexp1 == 4 and kqexp2 == 2:
						term = dim_gen(q2exp + 3,d1,dm)*(numk21**2*numk22 + 4*(k1dotk2(numk21,numk22,ksum2))**2*numk21)/35
					elif kqexp1 == 3 and kqexp2 == 3:
						term = dim_gen(q2exp + 3,d1,dm)*(3*numk21*numk22*k1dotk2(numk21,numk22,ksum2) + 2*(k1dotk2(numk21,numk22,ksum2))**3)/35
					elif kqexp1 == 2 and kqexp2 == 4:
						term = dim_gen(q2exp + 3,d1,dm)*(numk22**2*numk21 + 4*(k1dotk2(numk21,numk22,ksum2))**2*numk22)/35

					# cases where kqexp == 8
					elif kqexp1 == 4 and kqexp2 == 4:
						term = dim_gen(q2exp + 4,d1,dm)*(3*numk21**2*numk22**2 + 24*numk21*numk22*k1dotk2(numk21,numk22,ksum2)**2 + 8*(k1dotk2(numk21,numk22,ksum2))**4)/315
					
					else:
						print('ERROR: case not considered', kqexp, q2exp, kqexp1, kqexp2)		
				else:
					# case where kqexp == 0
					term = dim_gen(q2exp,d1,dm)

				res_list += term*term_list2[i2]*term_list1[i1]*(numk21)**(k2exp1)*(numk22)**(k2exp2)
			
	return res_list

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpfr k1dotk2(mpfr k21, mpfr k22, mpfr ksum2):
	return (ksum2 - k21 - k22)/2.



@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc Ltrian(long n1, long d1, long n2, long d2, long n3, long d3, 
		   mpfr k21, mpfr k22, mpfr k23, mpc m1, mpc m2, mpc m3, long m1_ind, long m2_ind, long m3_ind):
	
	#note that using hash gives a bug in python version 3.7. Need to use version 3.8 or above
	arg_list_bin = (n1,d1,n2,d2,n3,d3,m1_ind,m2_ind,m3_ind)

	# check if value exists in cache
	cached_result = Ltrian_cache.get(arg_list_bin)
	if cached_result:
		return cached_result	
	#if arg_list_bin in Ltrian_cache:
	#	return Ltrian_cache[arg_list_bin]

	cdef mpc result
	if n1 == 0 and n2 == 0 and n3 == 0:
		result = TriaN(d1,d2,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind)
		Ltrian_cache[arg_list_bin] = result
		return result
	if d1 == 0 and n1 != 0:
		result = Ltrian(0,-n1,n2,d2,n3,d3,k21,k22,k23,mpc0,m2,m3, 0, m2_ind, m3_ind)
		Ltrian_cache[arg_list_bin] = result
		return result
	if d2 == 0 and n2 != 0:
		result = Ltrian(n1,d1,0,-n2,n3,d3,k21,k22,k23,m1,mpc0,m3, m1_ind, 0, m3_ind)
		Ltrian_cache[arg_list_bin] = result
		return result	
	if d3 == 0 and n3 != 0:
		result = Ltrian(n1,d1,n2,d2,0,-n3,k21,k22,k23,m1,m2,mpc0, m1_ind, m2_ind, 0)
		Ltrian_cache[arg_list_bin] = result
		return result
	if n1 > 0:
		result = Ltrian(n1-1,d1-1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind) - m1*Ltrian(n1-1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind)
		Ltrian_cache[arg_list_bin] = result
		return result
	if n2 > 0:
		result = Ltrian(n1,d1,n2-1,d2-1,n3,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind) - m2*Ltrian(n1,d1,n2-1,d2,n3,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind)
		Ltrian_cache[arg_list_bin] = result
		return result
	if n3 > 0:
		result = Ltrian(n1,d1,n2,d2,n3-1,d3-1,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind) - m3*Ltrian(n1,d1,n2,d2,n3-1,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind)
		Ltrian_cache[arg_list_bin] = result
		return result
	if n1 < 0 :
		result = (Ltrian(n1,d1-1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind) - Ltrian(n1+1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind))/m1
		Ltrian_cache[arg_list_bin] = result
		return result
	if n2 < 0 :
		result = (Ltrian(n1,d1,n2,d2-1,n3,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind) - Ltrian(n1,d1,n2+1,d2,n3,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind))/m2
		Ltrian_cache[arg_list_bin] = result
		return result
	if n3 < 0:
		result = (Ltrian(n1,d1,n2,d2,n3,d3-1,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind) - Ltrian(n1,d1,n2,d2,n3+1,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind))/m3
		Ltrian_cache[arg_list_bin] = result
		return result
	else:
		print("Error: case not considered in Ltrian")



# This recursion AFFECTS SPEED, overall time ~6 times slower
#@cython.boundscheck(False)
#@cython.wraparound(False)

#@lru_cache(None)
#def Ltrian(long n1, long d1, long n2, long d2, long n3, long d3, 
#		   mpfr k21, mpfr k22, mpfr k23, mpc m1, mpc m2, mpc m3):
#
#	if n1 == 0 and n2 == 0 and n3 == 0:
#		return TriaN(d1,d2,d3,k21,k22,k23,m1,m2,m3)
#	if d1 == 0 and n1 != 0:
#		return Ltrian(0,-n1,n2,d2,n3,d3,k21,k22,k23,mpc0,m2,m3)
#	if d2 == 0 and n2 != 0:
#		return Ltrian(n1,d1,0,-n2,n3,d3,k21,k22,k23,m1,mpc0,m3)
#	if d3 == 0 and n3 != 0:
#		return Ltrian(n1,d1,n2,d2,0,-n3,k21,k22,k23,m1,m2,mpc0)
#	if n1 > 0:
#		return Ltrian(n1-1,d1-1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3) - m1*Ltrian(n1-1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3)
#	if n2 > 0:
#		return Ltrian(n1,d1,n2-1,d2-1,n3,d3,k21,k22,k23,m1,m2,m3) - m2*Ltrian(n1,d1,n2-1,d2,n3,d3,k21,k22,k23,m1,m2,m3)
#	if n3 > 0:
#		return Ltrian(n1,d1,n2,d2,n3-1,d3-1,k21,k22,k23,m1,m2,m3) - m3*Ltrian(n1,d1,n2,d2,n3-1,d3,k21,k22,k23,m1,m2,m3)
#	if n1 < 0:
#		return (Ltrian(n1,d1-1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3) - Ltrian(n1+1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3))/m1
#	if n2 < 0:
#		return (Ltrian(n1,d1,n2,d2-1,n3,d3,k21,k22,k23,m1,m2,m3) - Ltrian(n1,d1,n2+1,d2,n3,d3,k21,k22,k23,m1,m2,m3))/m2
#	if n3 < 0:
#		return (Ltrian(n1,d1,n2,d2,n3,d3-1,k21,k22,k23,m1,m2,m3) - Ltrian(n1,d1,n2,d2,n3+1,d3,k21,k22,k23,m1,m2,m3))/m3
#	print("Error: case not considered in Ltrian")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex Ltrian_complex(long n1, long d1, long n2, long d2, long n3, long d3, 
		   mpfr k21, mpfr k22, mpfr k23, mpc m1, mpc m2, mpc m3,
		   long m1_ind, long m2_ind, long m3_ind):
	
	arg_list = (n1,d1,n2,d2,n3,d3,m1_ind,m2_ind,m3_ind)
	if arg_list in Ltrian_complex_cache:
		return Ltrian_complex_cache[arg_list]

	cdef double complex result = <double complex>Ltrian(n1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3,m1_ind,m2_ind,m3_ind)
	Ltrian_complex_cache[arg_list] = result
	return result


# BubMaster does not affect the speed 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc BubMaster(mpfr k2, mpc M1, mpc M2):	
	cdef mpc m1 = M1/k2
	cdef mpc m2 = M2/k2
	cdef int sign = 0
	cdef mpc bubmaster

	cdef mpc arglog0 = 1j*(m1 - m2 - 1) + 2*sqrt(m1)
	cdef mpc arglog1 = 1j*(m1 - m2 + 1)+ 2*sqrt(m2)

	if (arglog0).imag > 0 and (arglog1).imag < 0:
		sign = 1
	else:
		sign = 0
	bubmaster = SQRT_PI/sqrt(k2)*(1j*(log(arglog0)-log(arglog1))+2*PI*sign)
	return bubmaster


#some short useful functions for the Triangle Master integral

@cython.boundscheck(False)
cdef mpc Diakr(mpfr a, mpc b, mpc c):
	return b**2-4*a*c

@cython.boundscheck(False)
@cython.wraparound(False)
#cdef mpc Prefactor(mpfr a, mpc y1, mpc y2):
def Prefactor(mpfr a, mpc y1, mpc y2):
	#computes prefactor that shows up in Fint

	cdef mpfr y2re = y2.real
	cdef mpfr y1re = y1.real

	if abs(y2.imag) < CHOP_TOL and abs(y1.imag) < CHOP_TOL:
		if abs(y1re) >= CHOP_TOL and abs(y2re) >= CHOP_TOL: 
			return sqrt(-y1re)*sqrt(-y2re)/(sqrt(a*(y1re)*(y2re)))
		if abs(y1re) < CHOP_TOL and abs(y2re) >= CHOP_TOL:
			return sqrt(-y2re)/sqrt(-a*y2re)
		if abs(y1re) >= CHOP_TOL and abs(y2re) < CHOP_TOL:
			return sqrt(-y1re)/sqrt(-a*(y1re))
		if abs(y1re) < CHOP_TOL and abs(y2re) < CHOP_TOL:
			return 1/sqrt(a)

	elif abs(y2.imag) >= CHOP_TOL and abs(y1.imag) < CHOP_TOL:
		if abs(y1re) >= CHOP_TOL: 
			return sqrt(-y1re)*sqrt(-y2)/sqrt(a*y1re*y2)
		if abs(y1re) < CHOP_TOL:
			return sqrt(-y2)/sqrt(-a*y2)

	elif abs(y2.imag) < CHOP_TOL and abs(y1.imag) >= CHOP_TOL:
		if abs(y2re) > CHOP_TOL: 
			return sqrt(-y1)*sqrt(-y2re)/(sqrt(a*y1*y2re))
		if abs(y2re) < CHOP_TOL:
			return sqrt(-y1)/sqrt(-a*y1)
	else:
		# case where abs(y2.imag) >= CHOP_TOL and abs(y1.imag) >= CHOP_TOL
		return sqrt(-y1)*sqrt(-y2)/sqrt(a*y1*y2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc Antideriv(long double x, mpc y1, mpc y2, mpc x0):
	if almosteq(x0,y2,CHOP_TOL):
		# case where x0 = y2 = 0 or 1
		if almosteq(mpc(x),y2,CHOP_TOL):
			return 0
		return 2.*sqrt(x-y1)/(-x0+y1)/sqrt(x-y2)
	
	if abs(x0-y1) < CHOP_TOL:
		print('WARNING: switching var in Antideriv')
		#x0 = y2 = 0 or 1
		return Antideriv(x,y2,y1,x0)

	
	cdef mpc prefac = 2/(sqrt(-x0+y1)*sqrt(x0-y2))
	cdef mpc temp = sqrt(mpc(x)-y1)*sqrt(x0-y2)/sqrt(-x0+y1)	
	cdef mpc LimArcTan
	# print('temp', temp)
	if x == 1 and almosteq(mpc(1), y2, CHOP_TOL):
		LimArcTan = 1j * sqrt(-temp**2) * PI/(2*temp)
		return  prefac * LimArcTan
	if x == 0 and almosteq(mpc0, y2, CHOP_TOL):
		LimArcTan = sqrt(temp**2) * PI/(2*temp)
		return  prefac * LimArcTan

	return prefac*atan(temp/sqrt(x-y2))
	

#Calculation of the Triangle Master integral 

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc Fint(mpfr aa, mpc Y1, mpc Y2, mpc X0):

	cdef mpc y1 = Y1
	cdef mpc y2 = Y2
	cdef mpc x0 = X0

	# this is necessary because in gmpy2 sqrt(-1-0j) = -1j and not 1j
	if abs(y2.imag) < CHOP_TOL:
		y2 = mpc(y2.real)
	if abs(y1.imag) < CHOP_TOL:
		y1 = mpc(y1.real)
	if abs(x0.imag) < CHOP_TOL:
		x0 = mpc(x0.real)

	cdef mpfr rey1 = y1.real
	cdef mpfr imy1 = y1.imag
	cdef mpfr rey2 = y2.real
	cdef mpfr imy2 = y2.imag
	cdef mpfr rex0 = x0.real
	cdef mpfr imx0 = x0.imag

	cdef int numbranchpoints = 0, signx0 = 0, sign = 0
	cdef list xsol = [], xbranch = [], atanarglist = [], abscrit = [], recrit = [], derivcrit = []

	cdef mpfr c = imy1**2*imy2*rex0 - imy1*imy2**2*rex0-imx0**2*imy2*rey1 + imx0*imy2**2*rey1-imy2*rex0**2*rey1 + imy2*rex0*rey1**2 + imx0**2*imy1*rey2-imx0*imy1**2*rey2+imy1*rex0**2*rey2-imx0*rey1**2*rey2-imy1*rex0*rey2**2+imx0*rey1*rey2**2
	cdef mpfr a = imy1*rex0-imy2*rex0-imx0*rey1+imy2*rey1+imx0*rey2-imy1*rey2
	cdef mpfr b = -imx0**2*imy1 + imx0*imy1**2+imx0**2*imy2-imy1**2*imy2-imx0*imy2**2+imy1*imy2**2-imy1*rex0**2+imy2*rex0**2+imx0*rey1**2-imy2*rey1**2-imx0*rey2**2+imy1*rey2**2

	cdef Py_ssize_t i

# if x0 is real there will always be a crossing through i or -i, which gives a cut of pi/2 instead of pi
	cdef mpc cutx0 = mpc0

	if 0 < rex0 < 1 and abs(imx0) < CHOP_TOL:
		derivcritx0 = (y1 - y2)/2/sqrt(-(rex0-y1)**2)/(rex0-y2)		
		if derivcritx0.real < 0:
			signx0 = 1
		else:
			signx0 = -1	
		cutx0 = signx0*PI/(sqrt(-rex0+y1+0j)*sqrt(rex0-y2+0j))
	else:
		cutx0 = mpc0
		
# find remaining crossings of the imaginary axis

	if abs(a) < CHOP_TOL:
		if b != 0:
			xsol = [- c / b]
		else:
			xsol = []
	else:
		if b**2-4*a*c > 0:
			xsol = [(-b + sqrt(b**2-4*a*c))/(2*a),(-b - sqrt(b**2-4*a*c))/(2*a)]
		else:
			#case where there is no intersection of the real axis (includes double zero)
			xsol = []

	xsol = [x for x in xsol if x > CHOP_TOL and x < mpfr(1) - CHOP_TOL and not(almosteq(mpc(x),x0,CHOP_TOL))]
	
	cdef mpc cut = mpc0
	if len(xsol) > 0:

		atanarglist = [sqrt(x-y1)*sqrt(x0-y2)/(sqrt(-x0+y1)*sqrt(x-y2)) for x in xsol]
		abscrit = [abs(atanarg) for atanarg in atanarglist]
		recrit = [atanarg.real for atanarg in atanarglist]
	
		for i in range(len(xsol)):
			if abscrit[i] > 1 and abs(recrit[i])<CHOP_TOL:
				numbranchpoints += 1
				xbranch.append(xsol[i])

	if numbranchpoints == 1:
		derivcrit = [sqrt(x0-y2)/sqrt(-x0+y1)*(1/(2*sqrt(x-y1)*sqrt(x-y2)) -sqrt(x-y1)/(2*(x-y2)*sqrt(x-y2))) for x in xbranch]
		if derivcrit[0].real < 0:
			sign = 1
		else:
			sign = -1
		cut = sign*PI*2/(sqrt(-x0+y1)*sqrt(x0-y2))
	else:
		cut = mpc0

	cdef mpc prefac0 = mpc(Prefactor(aa,y1,y2))
	cdef mpc result = prefac0*(SQRT_PI/2.)*(cut + cutx0 + Antideriv(1,y1,y2,x0) - Antideriv(0,y1,y2,x0))

	return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc TrMxy(long double y, mpfr k21, mpfr k22, mpfr k23, mpc M1, mpc M2, mpc M3):

	cdef mpfr Num1 = 4*k22*y+2*k21-2*k22-2*k23
	cdef mpc Num0 = -4*k22*y+2*M2-2*M3+2*k22
	cdef mpfr DeltaR2 = -k21*y+k23*y-k23
	cdef mpc DeltaR1 = -M2*y+M3*y+k21*y-k23*y+M1-M3+k23
	cdef mpc DeltaR0 = M2*y-M3*y+M3
	cdef mpfr DeltaS2 = -k21**2+2*k21*k22+2*k21*k23-k22**2+2*k22*k23-k23**2
	cdef mpc DeltaS1 =-4*M1*k22-2*M2*k21+2*M2*k22+2*M2*k23+2*M3*k21+2*M3*k22-2*M3*k23-2*k21*k22+2*k22**2-2*k22*k23
	cdef mpc DeltaS0 =-M2**2+2*M2*M3-2*M2*k22-M3**2-2*M3*k22-k22**2

	cdef mpc DiakrS = sqrt(Diakr(DeltaS2, DeltaS1, DeltaS0))
	cdef mpc solS1 = (-DeltaS1+DiakrS)/2/DeltaS2
	cdef mpc solS2 = (-DeltaS1-DiakrS)/2/DeltaS2  

	cdef mpc cf2 = -(Num1*solS2+Num0)/DiakrS
	cdef mpc cf1 = (Num1*solS1+Num0)/DiakrS
		
	cdef mpc DiakrR = sqrt(Diakr(DeltaR2, DeltaR1, DeltaR0))
				  
	cdef mpc solR1 = ((-DeltaR1+DiakrR)/2)/DeltaR2     
	cdef mpc solR2 = ((-DeltaR1-DiakrR)/2)/DeltaR2 

	if abs(cf1) < CHOP_TOL:
		# neglect cf1
		return cf2*Fint(DeltaR2, solR1, solR2, solS2)
	elif abs(cf2) < CHOP_TOL:
		# neglect cf2
		return cf1*Fint(DeltaR2, solR1, solR2, solS1)
	else:
		return cf2*Fint(DeltaR2, solR1, solR2, solS2)+cf1*Fint(DeltaR2, solR1, solR2, solS1)

#@lru_cache(None)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc TriaMasterZeroMasses(mpfr k21, mpfr k22, mpfr k23):
	#case for triangle integrals where all masses vanish
	return mpc(PI*SQRT_PI/sqrt(k21)/sqrt(k22)/sqrt(k23))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef mpc TriaMaster(mpfr k21, mpfr k22, mpfr k23, mpc M1, mpc M2, mpc M3):
	#--- masses are squared
	if M1 == mpc0 and M2 == mpc0 and M3 == mpc0:
		return  TriaMasterZeroMasses(k21, k22, k23)
	
	return TrMxy(1, k21, k22, k23, M1, M2, M3)-TrMxy(0, k21, k22, k23,M1, M2, M3)

