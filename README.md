[![Build Status](https://travis-ci.org/lruthotto/KrylovMethods.jl.svg?branch=master)](https://travis-ci.org/lruthotto/KrylovMethods.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/tkn9ssik9n9vgdt2?svg=true)](https://ci.appveyor.com/project/lruthotto/krylovmethods-jl)
[![Coverage Status](https://coveralls.io/repos/lruthotto/KrylovMethods.jl/badge.svg)](https://coveralls.io/r/lruthotto/KrylovMethods.jl)




KrylovMethods.jl
=========================

Released under the [MIT License](https://github.com/lruthotto/KrylovMethods.jl/blob/master/LICENSE).

Simple and fast Julia implementation of Krylov subspace methods for linear systems.

## Goals and Guidelines

The main goal of this package is to derive simple and fast implementation of the most useful Krylov subspace methods. 

Further goals are
- **speed**: where try to minimize allocation costs and maximize the use of BLAS routines,
- **memory efficiency**: storing temporary variables and re-allocations are avoided,
- **generality**: where possible complex systems are supported,
- **reliability**: unit tests are provided.






