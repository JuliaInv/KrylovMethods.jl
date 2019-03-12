
@testset "bicgstb" begin
	@testset "real matrix" begin
		A  = sprandn(100,100,.1) + 10*I
		D  = Array(diag(A))
		Af = x -> A*x 
		M  = x -> D.\x
		rhs = randn(100)
		
		# test flag for early stopping
		xt = bicgstb(A,rhs,tol=1e-6,maxIter=3,out=2,storeInterm=true)
		@test xt[2]==-1
		
		# test handling of zero rhs
		xt = bicgstb(A,zeros(size(A,2)),tol=1e-6,maxIter=3,out=2)
		@test  all(xt[1].==0)
		@test xt[2]==-9
		
		
		x0 = bicgstb(Matrix(A),rhs,tol=1e-6)
		x1 = bicgstb(A,rhs,tol=1e-6,out=1)
		x2 = bicgstb(Af,rhs,tol=1e-6)
		x3 = bicgstb(Af,rhs,tol=1e-6,maxIter=100,x=randn(size(rhs)))
		x5 = bicgstb(Af,rhs,tol=1e-6,maxIter=100,M1=M)
		
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
		A  = sprandn(100,100,.1) + 10*I + im*(sprandn(100,100,.1) + 10*I )
		D  = Array(diag(A))
		Af = x -> A*x 
		M  = x -> D.\x
		rhs = randn(100) + 1im * randn(100)
		
		x1 = bicgstb(A,rhs,tol=1e-6)
		x2 = bicgstb(Af,rhs,tol=1e-6)
		x3 = bicgstb(Af,rhs,tol=1e-6,maxIter=200,x=randn(size(rhs))+im*randn(size(rhs)))
		x4 = bicgstb(Af,rhs,tol=1e-6,maxIter=200,M1=M)
		
		@test norm(A*x1[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x2[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x3[1]-rhs)/norm(rhs) < 1e-6
		@test norm(A*x4[1]-rhs)/norm(rhs) < 1e-6
		@test norm(x2[1]-x1[1])/norm(x1[1]) < 1e-12
		@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-5

	end
end