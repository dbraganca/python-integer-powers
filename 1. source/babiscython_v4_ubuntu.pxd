from gmpy2 cimport *

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

cdef mpc Ltrian(long, long, long, long, long, long, mpfr, mpfr, mpfr, mpc, mpc, mpc, long, long, long, dict)

#cdef double complex Ltrian_complex(long, long, long, long, long, long, mpfr, mpfr, mpfr, mpc, mpc, mpc, long, long, long)