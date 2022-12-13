# Analytical integration of EFTofLSS loop corrections

This project was created to efficiently evaluate ETFofLSS loop corrections up to the 1-loop bispectrum. 
This technique uses a decomposition of the linear power spectrum into analytical function of $k^2$, and proceeds to evaluate the integrals using recursion relations.

## Outline

- [1. source/](1.%20source/): where all the computation scripts are.
- [3. Ctabs/](3.%20Ctabs/): the exponent tables and the $k$'s and/or triangles to evaluate are here.

## Installation

The main script is `1. source/babiscython_v4_ubuntu.pyx` and is written in Cython. 
It calculates the function $L$.
To compile it, one needs to run `1. source/setup_babiscython_ubuntu.py` the following way:
```
python setup_babiscython_ubuntu.py build_ext --inplace
```

After this, one needs to compile the script `1. source/Jfunc_cython_v4.pyx` which calculates the $J$ function from the $L$ function.
To compile it, one needs to run 
```
python setup_jfunc.py build_ext --inplace
```

The file `1. source/config.py` contains the cache for memoization of the $L$-function.

## Example of usage

The remaining files calculate individual loops.
For example, running 
```
python B222_bias.py
```

Calculates the bias-decomposed $J$-function for a given set of triangles.

The notebook `1. source/Test and run functions.ipynb` provides other examples of usage.