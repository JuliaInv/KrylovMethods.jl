using SimpleIterativeSolvers
using Base.Test

println("=== Testing CLGS ===")

A = sprandn(100,10,.2)
Af(x,flag) = (flag=='F') ? A*x : A'*x

rhs = randn(100)

xgt = (A'*A)\(A'*rhs)

xt  = cgls(A,rhs,1e-20)
xt2  = cgls(A,rhs,1e-20,100,randn(size(xt[1])))
xf  = cgls(Af,rhs,1e-20)
Xt  = cgls(A,rhs,1e-20,100,[],1)

@test norm(xgt-xt[1])/norm(xgt) < 1e-6
@test norm(xt[1]-xf[1])/norm(xf[1]) < 1e-15
@test norm(Xt[1][:,end]-xt[1])/norm(xt[1]) < 1e-15

println("=== CGLS: All tests passed. ===")