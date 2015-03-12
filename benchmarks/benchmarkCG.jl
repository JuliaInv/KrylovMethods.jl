using KrylovMethods
import IterativeSolvers
include("../test/getDivGrad.jl")

println("=== Benchmark CG ===")
# CG: test sparse Laplacian
N = (24,48,64,96,128)
xCG = []; xCGmf = []; xIS = [];
for n=N
	
	A = getDivGrad(n,n,n)
	rhs = zeros(n,n,n)
	rhs[round(Int64,n/8):n-round(Int64,n/8),round(Int64,n/8):n-round(Int64,n/8),round(Int64,n/8):n-round(Int64,n/8)] = 1
	rhs[round(Int64,n/4):n-round(Int64,n/4),round(Int64,n/4):n-round(Int64,n/4),round(Int64,n/4):n-round(Int64,n/4)] = 10;
	rhs[round(Int64,n/3):n-round(Int64,n/3),round(Int64,n/3):n-round(Int64,n/3),round(Int64,n/3):n-round(Int64,n/3)] = 100;
	rhs = vec(rhs)
	tolCG = 1e-10
	
	
	tic()
	Af1 = x-> A*x
	xCG,flagCG,relresCG,iterCG,resvecCG       = KrylovMethods.cg(Af1,rhs,tol=tolCG,maxIter=100,out=1)
	timeAx = toq()
	
	temp = zeros(size(A,1))
	Af2 = x-> At_mul_B!(1.0,A,x,0.0,temp)
	tic()
	xCGmf,flagCG,relresCG,iterCG,resvecCG       = KrylovMethods.cg(Af2,rhs,tol=tolCG,maxIter=100,out=1)
	timeAmulB = toq()
	
	tic()
	xIS, = IterativeSolvers.cg(A,rhs,tol=1e-10,maxiter=100)
	timeIS = toq()
	
	println(@sprintf("n=%d\ttime(A*x)=%1.3f time(A_mul_B!)=%1.3f speedup=%1.3f",n, timeAx, timeAmulB, timeAx/timeAmulB ))
	println(@sprintf("n=%d\ttime(IterativeSolvers)=%1.3f time(KrylovMethods)=%1.3f speedup=%1.3f", n, timeAmulB,timeIS, timeIS/timeAmulB ))

end