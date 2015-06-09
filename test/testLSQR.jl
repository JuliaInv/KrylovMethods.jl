using KrylovMethods
using LinearOperators
using MatrixDepot
using Base.Test

println("=== Testing LSQR ===")

# test with sparse random matrix
A   = sprandn(100,10,.2)
rhs = randn(100)
xgt = full(A)\rhs
Af(flag,x,a=0.0,v=0.0) = (flag=='F') ? A*x +a*v : A'*x+a*v

x1  = lsqr(LinearOperator(A),rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10)
x2  = lsqr(A,rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10)
x3  = lsqr(Af,rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10)

@test norm(xgt-x1[1])/norm(xgt) < 1e-6
@test norm(xgt-x2[1])/norm(xgt) < 1e-6
@test norm(xgt-x3[1])/norm(xgt) < 1e-6
@test norm(x1[1]-x2[1])/norm(x1[1]) < 1e-12
@test norm(x2[1]-x3[1])/norm(x3[1]) < 1e-12

# test with dense hilbert matrix
A   = matrixdepot("hilb",10,3)
rhs = randn(size(A,1))
xgt = A\rhs

x1  = lsqr(LinearOperator(A),rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10)
x2  = lsqr(A,rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10)

@test norm(xgt-x1[1])/norm(xgt) < 1e-6
@test norm(xgt-x2[1])/norm(xgt) < 1e-6
@test norm(x1[1]-x2[1])/norm(x1[1]) < 1e-12
conA = norm(vec(A))*norm(vec(pinv(A)))
@test norm(conA-x1[3][end,end])/conA < 1e-12
@test norm(conA-x2[3][end,end])/conA < 1e-12

println("=== LSQR: All tests passed. ===")