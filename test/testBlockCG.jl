using KrylovMethods
using Test
using LinearOperators
include("getDivGrad.jl")

@testset "blockCG" begin
# small full system
A = [4.0 1; 1 4]
rhs = randn(2,2)
X,flag,relres,iter,resvec = blockCG(A,rhs,tol=1e-15,out=2,storeInterm=true)
@test norm(A*X[:,:,end]-rhs)/norm(rhs) <= 1e-14

# test message and flag when stopping early
X,flag,relres,iter,resvec = blockCG(A,randn(2,2),tol=1e-30,out=2,maxIter=1)
@test flag==-1

# test for zero right hand-side
X,flag = blockCG(A,zeros(2,2),out=2)
@test flag == -9
@test all(X.==0)

A      = getDivGrad(8,8,8)
rhs    = randn(size(A,1),10)
X,flag,relres,iter,resvec = blockCG(A,rhs,tol=1e-3,out=2,maxIter=200,storeInterm=false,ortho=true);
@test norm(A*X-rhs)/norm(rhs) < 1.5e-3
X,flag,relres,iter,resvec = blockCG(A,rhs,tol=1e-3,out=2,maxIter=200,storeInterm=false,ortho=false);
@test norm(A*X-rhs)/norm(rhs) < 1.5e-3


end