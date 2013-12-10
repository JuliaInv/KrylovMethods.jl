SimpleIterativeSolvers.jl
=========================

&copy; 2013 [Eldad Haber](http://www.math.ubc.ca/~haber) and [Lars Ruthotto](http://www.eos.ubc.ca/about/researcher/L.Ruthotto.html). Released under the [MIT License](https://github.com/lruthotto/SimpleIterativeSolvers.jl/blob/master/LICENSE).

Simple and fast Julia implementation of iterative solvers for linear systems.

## Goals and Guidelines

The main goal of this package is to derive simple and fast implementation of iterative linear solvers. 

For the sake of simplicity we want to
- avoid declaration of our own types,
- avoid using greek letters in source codes,
- provide readable and well-documented code.

Further goals are
- **speed**: where possible matrix-free implementations are given,
- **memory efficiency**: storing temporary variables and re-allocations are avoided,
- **generality**: where possible complex systems are supported,
- **reliability**: unit tests are provided.






