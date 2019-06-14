@testset "CLGS" begin

# test with sparse matrix
A   = sprandn(100,10,.2)
Af = (x,flag) -> ((flag=='F') ? A*x : A'*x)
rhs = randn(100)

# test early stopping behaviour
tt = cgls(Af,rhs,tol=1e-20,maxIter=2,out=2,storeInterm=true)
@test tt[2]==-1

# test behaviour for zero rhs
tt = cgls(Af,zeros(size(A,1)),tol=1e-20,maxIter=2,out=2)
@test tt[2]==-9
@test all(tt[1].==0)


xgt = Matrix(A)\rhs
xt  = cgls(LinearOperator(A),rhs,tol=1e-20)
xt2 = cgls(Af,rhs,tol=1e-20,maxIter=100,x=randn(size(xt[1])))
Xt  = cgls(A,rhs,tol=1e-20,maxIter=100,storeInterm=true,out=1)

@test norm(xgt-xt[1])/norm(xgt) < 1e-6
@test norm(xgt-xt2[1])/norm(xgt) < 1e-6
@test norm(Xt[1][:,end]-xt[1])/norm(xt[1]) < 1e-15

# test with dense hilbert matrix
# A   = matrixdepot("hilb",10,3)
# rhs = randn(size(A,1))
# xgt = A\rhs
#
# x1  = cgls(LinearOperator(A),rhs,tol=1e-10,maxIter=10)
# x2  = cgls(A,rhs,tol=1e-10,maxIter=10)
#
# @test norm(xgt-x1[1])/norm(xgt) < 1e-6
# @test norm(xgt-x2[1])/norm(xgt) < 1e-6
# @test norm(x1[1]-x2[1])/norm(x1[1]) < 1e-12

end
