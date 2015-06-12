using KrylovMethods
using Base.Test
using MatrixDepot


println("=== Testing Lanczos Bidiagonalization ===")
A = sprandn(10,10,.4) + speye(10)
p = randn(10)
k = 9

U,B,V = lanczosBidiag(A,p,k)

@test norm(full(U'*A*V-B))/norm(full(A)) < 1e-15


println("=== Testing Lanczos Tridiagonalization ===")

A   = matrixdepot("rosser",4)
Af  = x -> A*x
T1,V1 = lanczosTridiag(A,ones(size(A,1)),size(A,1),tol=1e-10,doReorth=true)
T2,V2 = lanczosTridiag(Af,ones(size(A,1)),size(A,1),tol=1e-10,doReorth=false)
sT1  = svd(full(T1))[2]
sT2  = svd(full(T2))[2]
sA   = svd(full(A))[2]

@test abs(sT1[1]-sA[1])/sA[1] < 1e-4
@test abs(sT2[1]-sA[1])/sA[1] < 1e-4
B = V1'*V1
@test norm(B - diagm(diag(B))) < 1e-10
@test norm(diag(B)- ones(size(B,1))) < 1e-10
println("=== Lanczos: All tests passed. ===")
