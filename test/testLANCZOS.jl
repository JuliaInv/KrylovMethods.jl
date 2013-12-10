using SimpleIterativeSolvers
using Base.Test

println("=== Testing Lanczos Bidiagonalization ===")

A = sprandn(10,10,.4) + speye(10)
p = randn(10)
k = 9

U,B,V = lanczosBidiag(A,p,k)

@test norm(full(U'*A*V-B))/norm(full(A)) < 1e-15

println("=== Lanczos: All tests passed. ===")
