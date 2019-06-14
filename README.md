[![Build Status](https://travis-ci.org/JuliaInv/KrylovMethods.jl.svg?branch=master)](https://travis-ci.org/JuliaInv/KrylovMethods.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/rcwxarativdkwbwp?svg=true)](https://ci.appveyor.com/project/lruthotto/krylovmethods-jl-qowll)
[![Coverage Status](https://coveralls.io/repos/github/JuliaInv/KrylovMethods.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaInv/KrylovMethods.jl?branch=master)




KrylovMethods.jl
=========================

Released under the [MIT License](https://github.com/lruthotto/KrylovMethods.jl/blob/master/LICENSE).

Simple and fast Julia implementation of Krylov subspace methods for linear systems.

## Goals and Guidelines

The main goal of this package is to derive simple and fast implementation of the most useful Krylov subspace methods.

Our main objectives are:
- **speed**: where try to minimize allocation costs and maximize the use of BLAS routines,
- **memory efficiency**: storing temporary variables and re-allocations are avoided,
- **generality**: where possible complex systems are supported,
- **reliability**: unit tests are provided.
