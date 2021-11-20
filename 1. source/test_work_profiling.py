import numpy as np
import os
import sys
from functools import lru_cache
import pandas as pd
from Jfunc_cython_v4 import computeJ as J
from babiscython_v4_ubuntu import Ltrian as L
import gmpy2 as gm
from gmpy2 import *
import time

kpeak1 = -0.034
# kpeak2 = 0.006
# kpeak3 = 0.000076
kpeak2 = -0.001
kpeak3 = -0.000076
kpeak4 = -0.0000156
kuv1 = 0.069
kuv2 = 0.0082
kuv3 = 0.0013
kuv4 = 0.0000135
kpeak1 = mpfr(str(kpeak1))
kpeak2 = mpfr(str(kpeak2))
kpeak3 = mpfr(str(kpeak3))
kpeak4 = mpfr(str(kpeak4))
kuv1 = mpfr(str(kuv1))
kuv2 = mpfr(str(kuv2))
kuv3 = mpfr(str(kuv3))
kuv4 = mpfr(str(kuv4))

mass1 = -kpeak1 - 1j*kuv1
mass1conj = -kpeak1 + 1j*kuv1
mass2 = -kpeak2 - 1j*kuv2
mass2conj = -kpeak2 + 1j*kuv2
mass3 = -kpeak3 - 1j*kuv3
mass3conj = -kpeak3 + 1j*kuv3
mass4 = -kpeak4 - 1j*kuv4
mass4conj = -kpeak4 + 1j*kuv4


k21 = mpfr(1.5)
k22 = mpfr(1.4)
k23 = mpfr(1.3)

n1 = -4
n2 = 2
n3 = 0

d1=2
d2=2
d3=2

m1 = mass1
m2 = mass2
m3 = mass3

start_time = time.time()
print("J",n1,n2,n3, J(n1,n2,n3,0,0,4, k21, k22, k23))
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
print("L",n1,d1,n2,d2,n3,d3,L(n1,d1,n2,d2,n3,d3,k21,k22,k23,m1,m2,m3))
print("--- %s seconds ---" % (time.time() - start_time))

