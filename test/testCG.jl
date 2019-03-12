
@testset "CG" begin
# small full system
A = [4.0 1; 1 4]
rhs = [2.0;2]
x,flag,relres,iter,resvec = cg(A,rhs,tol=1e-15,out=2,storeInterm=true)
@test norm(A*x[:,end]-rhs)/norm(rhs) <= 1e-15

# test message and flag when stopping early
x,flag,relres,iter,resvec = cg(A,randn(2),tol=1e-15,out=2,maxIter=1)
@test flag==-1

# test for zero right hand-side
x,flag = cg(A,zeros(2),out=2)
@test flag == -9
@test all(x.==0)

# negative definite matrix
b = [1 -1.0]
KKT = [A b'; b 0]
x,flag, = cg(KKT,randn(3),out=1)
@test flag==-2


# CG: test sparse Laplacian
A = getDivGrad(8,8,8)
Alinop = LinearOperator(A)
L = tril(A)
D = Array(diag(A))
U = triu(A)
n = size(A,1)
JAC = (x) -> D.\x
SGS = (x) -> L\(D.*(U\x))

rhs = randn(size(A,1))
tolCG = 1e-5
# tests with A being matrix
xCG,flagCG,relresCG,iterCG,resvecCG       = cg(A,rhs,tol=tolCG,maxIter=100)
xJAC,flagJAC,relresJAC,iterJAC,resvecJAC  = cg(A,rhs,tol=tolCG,maxIter=100,M=JAC,out=1)
xSGS,flagSGS,relresSGS,iterSGS,resvecSGS  = cg(A,rhs,tol=tolCG,maxIter=100,M=SGS)
# tests with A being function
xCGmf,flagCG,relresCG,iterCG,resvecCG       = cg(Alinop,rhs,tol=tolCG,maxIter=100)
xJACmf,flagJAC,relresJAC,iterJAC,resvecJAC  = cg(Alinop,rhs,tol=tolCG,maxIter=100,M=JAC)
xSGSmf,flagSGS,relresSGS,iterSGS,resvecSGS  = cg(Alinop,rhs,tol=tolCG,maxIter=100,M=SGS)
# tests with random starting guess
xCGr,flagCGr,relresCGr,iterCGr,resvecCGr       = cg(Alinop,rhs,tol=tolCG,maxIter=100,x=randn(size(rhs)))
xJACr,flagJACr,relresJACr,iterJACr,resvecJACr  = cg(Alinop,rhs,tol=tolCG,maxIter=100,M=JAC,x=randn(size(rhs)))
xSGSr,flagSGSr,relresSGSr,iterSGSr,resvecSGSr  = cg(A,rhs,tol=tolCG,maxIter=100,x=randn(size(rhs)))

# test relative residuals
@test norm(A*xCG-rhs)/norm(rhs) <= tolCG
@test norm(A*xSGS-rhs)/norm(rhs) <= tolCG
@test norm(A*xJAC-rhs)/norm(rhs) <= tolCG
@test norm(A*xCGmf-rhs)/norm(rhs) <= tolCG
@test norm(A*xSGSmf-rhs)/norm(rhs) <= tolCG
@test norm(A*xJACmf-rhs)/norm(rhs) <= tolCG
@test norm(A*xCGr-rhs)/norm(rhs) <= tolCG*10
@test norm(A*xSGSr-rhs)/norm(rhs) <= tolCG*10
@test norm(A*xJACr-rhs)/norm(rhs) <= tolCG*10
# preconditioners should at least not increase number of iter
@test iterJAC==iterCG
@test iterSGS<=iterJAC

end