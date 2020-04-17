
@testset "blockBiCGSTB" begin
	@testset "real matrix" begin
		nrhs = 4;
		n = 100
		A  = sprandn(n,n,.1) + 10*I
		D  = Array(diag(A))
		Af = x ->  A*x 
		M  = x ->  D.\x
		rhs = randn(n,nrhs)
		
		# test flag for early stopping
		xt = blockBiCGSTB(A,rhs,tol=1e-6,maxIter=3,out=2)
		@test xt[2]==-1
		
		# test handling of zero rhs
		xt = blockBiCGSTB(A,zeros(n,nrhs),tol=1e-6,maxIter=3,out=2)
		@test  all(xt[1].==0)
		@test xt[2]==-9
		
		
		x0 = blockBiCGSTB(Matrix(A),rhs,tol=1e-6)
		x1 = blockBiCGSTB(A,rhs,tol=1e-6,out=1)
		x2 = blockBiCGSTB(Af,rhs,tol=1e-6)
		x3 = blockBiCGSTB(Af,rhs,tol=1e-6,maxIter=100,x=randn(size(rhs)))
		x5 = blockBiCGSTB(Af,rhs,tol=1e-6,maxIter=100,M1=M)
		
		@test norm(A*x0[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x1[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x2[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x3[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x5[1]-rhs)/norm(rhs) < 1e-6
		@test norm(x2[1]-x1[1])/norm(x1[1]) < 1e-12
		@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-5
		@test norm(x0[1]-x1[1])/norm(x1[1]) < 1e-5
	end
	@testset "complex matrix" begin
		nrhs = 4;
		n = 100
		A  = sprandn(n,n,.1) + 10*I + im*(sprandn(n,n,.1) + 10*I )
		D  = Array(diag(A))
		Af = x -> A*x 
		M  = x -> D.\x
		rhs = randn(n,nrhs) + 1im * randn(n,nrhs)
		
		x1 = blockBiCGSTB(A,rhs,tol=1e-6)
		x2 = blockBiCGSTB(Af,rhs,tol=1e-6)
		x3 = blockBiCGSTB(Af,rhs,tol=1e-6,maxIter=200,x=randn(size(rhs))+im*randn(size(rhs)))
		x4 = blockBiCGSTB(Af,rhs,tol=1e-6,maxIter=200,M1=M)
		
		@test norm(A*x1[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x2[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x3[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x4[1]-rhs)/norm(rhs) < 1e-6
		@test norm(x2[1]-x1[1])/norm(x1[1]) < 1e-12
		@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-5
	end
end
