using KrylovMethods
import IterativeSolvers
using ProfileView
include("../test/getDivGrad.jl")

println("=== Benchmark CG ===")
# CG: test sparse Laplacian
N = (24,48,64,96,128)
xCG = []; xCGmf = []; xIS = [];
for n=N
	
	A = getDivGrad(n,n,n)
	rhs = zeros(n,n,n)
	rhs[round(Int,n/8):n-round(Int,n/8),round(Int,n/8):n-round(Int,n/8),round(Int,n/8):n-round(Int,n/8)] = 1
	rhs[round(Int,n/4):n-round(Int,n/4),round(Int,n/4):n-round(Int,n/4),round(Int,n/4):n-round(Int,n/4)] = 10;
	rhs[round(Int,n/3):n-round(Int,n/3),round(Int,n/3):n-round(Int,n/3),round(Int,n/3):n-round(Int,n/3)] = 100;
	rhs = vec(rhs)
	tolCG = 1e-10
	
	
	tic()
	xCG,flagCG,relresCG,iterCG,resvecCG       = KrylovMethods.cg(A,rhs,tol=tolCG,maxIter=100,out=1)
	timeAx = toq()
	
	
	tic()
	xIS, = IterativeSolvers.cg(A,rhs,tol=1e-10,maxiter=100)
	timeIS = toq()
	
	println(@sprintf("n=%d\ttime(IterativeSolvers)=%1.3f time(KrylovMethods)=%1.3f speedup=%1.3f", n, timeIS,timeAx, timeIS/timeAx ))

end