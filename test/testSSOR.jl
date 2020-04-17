
@testset "SSOR" begin
	@testset "small full system" begin
		# CG: test sparse Laplacian
		A = getDivGrad(32,32,32)
		rhs = randn(size(A,1))
		tol = 1e-2

		# tests with A being matrix
		x1,flag1,relres1,iter1,resvec1       = ssor(A,rhs,tol=tol,maxIter=100)
		x2,flag2,relres2,iter2,resvec2       = ssor(A,rhs,tol=tol,maxIter=100,omega=2/3,out=1)
		x3,flag3,relres3,iter3,resvec3     = ssor(A,rhs,tol=tol,maxIter=100,omega=1/2,out=1,storeInterm=true)

		# test relative residuals
		@test norm(A*x1-rhs)/norm(rhs) <= tol
		@test norm(A*x2-rhs)/norm(rhs) <= tol
		@test norm(A*x3[:,end]-rhs)/norm(rhs) <= tol
	end

	@testset "preconditioner" begin
		omega = 1.2;
		A = getDivGrad(32,32,32)
		rhs = randn(size(A,1))
		d = omega./Array(diag(A));
		x = zeros(length(rhs)) # pre allocation for the preconditioner result.
		PC = r -> (x[:].=0.0; return ssorPrecTrans!(A,x,r,d));
		y = KrylovMethods.cg(A,rhs,tol=1e-12,maxIter=200,M=PC,out=1)[1]
		@test norm(A*y-rhs)/norm(rhs) <= 1e-12
	end

	@testset "complex " begin

		# CG: test sparse Laplacian
		A = getDivGrad(32,32,32)
		A += 1im*I
		rhs = randn(size(A,1)) + 1im * randn(size(A,1))
		tol = 1e-2

		#
		x0,flag0,relres0,iter0,resvec0       = ssor(A,rhs,tol=tol,maxIter=3,out=2)
		@test flag0==-1
		@test iter0==3
		@test length(resvec0)==3

		# tests with A being matrix
		x1,flag1,relres1,iter1,resvec1       = ssor(A,rhs,tol=tol,maxIter=100)
		x2,flag2,relres2,iter2,resvec2       = ssor(A,rhs,tol=tol,maxIter=100,omega=2/3,out=1)
		x3,flag3,relres3,iter3,resvec3       = ssor(A,rhs,tol=tol,maxIter=100,omega=1/2,out=1)

		# test relative residuals
		@test norm(A*x1-rhs)/norm(rhs) <= tol
		@test norm(A*x2-rhs)/norm(rhs) <= tol
		@test all(diff(resvec1).<0)
		@test all(diff(resvec2).<0)
		@test all(diff(resvec3).<0)

	end
end
