

@testset "gmres" begin
	@testset "real matrix" begin
		A  = sprandn(100,100,.1) + 10*I
		n  = size(A,2)
		D  = Array(diag(A))
		M2 = x -> D.\x
		rhs = randn(100)
		tol = 1e-6;
		
		# test printing and behaviour for early stopping
		xtt = gmres(A,rhs ,3,tol=1e-10,maxIter=3,out=2,storeInterm=true)
		@test xtt[2]==-1
		
		# test behaviour for zero rhs
		xtt = gmres(A,0*rhs,5,tol=tol,maxIter=10,out=2)
		@test xtt[2]==-9
		@test all(xtt[1].==0)
		@test length(xtt[1])==100
		@test eltype(xtt[1])==eltype(rhs)
		
		
		
		x1 = gmres(A,rhs ,5,tol=tol,maxIter=100,out=1)
		x3 = gmres(A,rhs,5,tol=tol,maxIter=100,x=randn(size(rhs)))
		x4 = gmres(A,rhs,5,tol=tol,maxIter=100,M=M2)
		
		@test norm(A*x1[1]-rhs)/norm(rhs) < tol
		@test norm(A*x3[1]-rhs)/norm(rhs) < tol
		# @test norm(A*x4[1]-rhs)/norm(rhs) < tol
		@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-5
	end

	@testset "complex matrix" begin
		A  = sprandn(100,100,.1) + 10*I + im*(sprandn(100,100,.1) + 10*I )
		D  = Array(diag(A))
		M3 = x -> D.\x
		rhs = complex(randn(100))
		tol = 1e-6;
		
		# test behaviour for zero rhs
		xtt = gmres(A,0*rhs,5,tol=tol,maxIter=10,out=2)
		@test xtt[2]==-9
		@test all(xtt[1].==0)
		@test length(xtt[1])==100
		@test eltype(xtt[1])==eltype(rhs)
		
		
		x1 = gmres(A,rhs ,5,tol=tol,maxIter=100)
		x3 = gmres(A,rhs,5,tol=tol,maxIter=100,x=randn(size(rhs)))
		x4 = gmres(A,rhs,5,tol=tol,maxIter=100,M=M3)
		
		@test norm(A*x1[1]-rhs)/norm(rhs) < tol
		@test norm(A*x3[1]-rhs)/norm(rhs) < tol
		@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-5
	end
end