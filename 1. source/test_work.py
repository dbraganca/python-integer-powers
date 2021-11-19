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


k21 = mpfr(1.5)
k22 = mpfr(1.4)
k23 = mpfr(1.3)

n1 = -4
n2 = 2
n3 = 0

start_time = time.time()
print(n1,n2,n3, J(n1,n2,n3,0,0,4, k21, k22, k23))
print("--- %s seconds ---" % (time.time() - start_time))
