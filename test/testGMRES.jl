using SimpleIterativeSolvers
using Base.Test

println("=== Testing gmres for real matrix === ")
A  = sprandn(100,100,.1) + 10*speye(100)
D  = diag(A)
Af(x) = A*x 
M(x)  = D.\x
rhs = randn(100)
tol = 1e-6;

x1 = gmres(A,rhs ,5,tol=tol,maxIter=100)
x2 = gmres(Af,rhs,5,tol=tol)
x3 = gmres(Af,rhs,5,tol=tol,maxIter=100,x=randn(size(rhs)))
x4 = gmres(Af,rhs,5,tol=tol,maxIter=100,M=M)

@test norm(A*x1[1]-rhs)/norm(rhs) < tol
@test norm(A*x2[1]-rhs)/norm(rhs) < tol
@test norm(A*x3[1]-rhs)/norm(rhs) < tol
# @test norm(A*x4[1]-rhs)/norm(rhs) < tol
@test norm(x2[1]-x1[1])/norm(x1[1]) < 1e-12
@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-5

println("=== Testing gmres for complex matrix === ")
A  = sprandn(100,100,.1) + 10*speye(100) + im*(sprandn(100,100,.1) + 10*speye(100) )
D  = diag(A)
Af(x) = A*x 
M(x)  = D.\x
rhs = complex(randn(100))

x1 = gmres(A,rhs ,5,tol=tol,maxIter=100)
x2 = gmres(Af,rhs,5,tol=tol)
x3 = gmres(Af,rhs,5,tol=tol,maxIter=100,x=randn(size(rhs)))
x4 = gmres(Af,rhs,5,tol=tol,maxIter=100,M=M)

@test norm(A*x1[1]-rhs)/norm(rhs) < tol
@test norm(A*x2[1]-rhs)/norm(rhs) < tol
@test norm(A*x3[1]-rhs)/norm(rhs) < tol
# @test norm(A*x4[1]-rhs)/norm(rhs) < tol
@test norm(x2[1]-x1[1])/norm(x1[1]) < 1e-12
@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-5

println("=== GMRES: All tests passed. ===")