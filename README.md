KrylovMethods.jl
=========================

&copy; 2014 [Lars Ruthotto](http://www.eos.ubc.ca/about/researcher/L.Ruthotto.html). Released under the [MIT License](https://github.com/lruthotto/KrylovMethods.jl/blob/master/LICENSE).

Simple and fast Julia implementation of Krylov subspace methods for linear systems.

## Goals and Guidelines

The main goal of this package is to derive simple and fast implementation of the most useful Krylov subspace methods. 

For the sake of simplicity aim to
- avoid declaration of our own types,
- avoid using greek letters in source codes,
- provide tested, readable and well-documented code.

Further goals are
- **speed**: where try to minimize allocation costs and maximize the use of BLAS routines,
- **memory efficiency**: storing temporary variables and re-allocations are avoided,
- **generality**: where possible complex systems are supported,
- **reliability**: unit tests are provided.






