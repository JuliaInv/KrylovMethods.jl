
println("*******************************************************")
@testset "qmr" begin
	@testset "real matrix" begin
		A  = sprandn(100,100,.1)
		A  = A + SparseMatrixCSC(100.0I, 100, 100)
		n  = size(A,2)
		D  = diag(A)
		M2 = x -> Vector(D.\x)
		rhs = randn(100)
		tol = 1e-2;
		
		# test printing and behaviour for early stopping
		xtt = qmr(A,rhs,tol=1e-12,maxIter=3,out=2)
		@test xtt[2]==-1
		
		# test behaviour for zero rhs
		xtt = qmr(A,0*rhs,tol=tol,maxIter=10,out=2)
		@test xtt[2]==-9
		@test all(xtt[1].==0)
		@test length(xtt[1])==100
		@test eltype(xtt[1])==eltype(rhs)

		
		x1 = qmr(A,rhs ,tol=tol,maxIter=100,out=1)
		x3 = qmr(A,rhs,tol=tol,maxIter=100,x=randn(size(rhs)))
		x4 = qmr(A,rhs,tol=tol,maxIter=100,M=M2)
		
		@test norm(A*x1[1]-rhs)/norm(rhs) < tol
		@test norm(A*x3[1]-rhs)/norm(rhs) < tol
		@test norm(A*x4[1]-rhs)/norm(rhs) < tol
		@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-2
	end

	@testset "complex matrix" begin
		A = sprandn(100,100,.1) + 1im*(sprandn(100,100,.1));
		A  = A + SparseMatrixCSC(100.0I, 100, 100)
		D  = Vector(diag(A))
		M3 = x -> x./D
		rhs = complex(randn(100))
		tol = 1e-3;
		
		# test behaviour for zero rhs

		xtt = qmr(A,0.0*rhs,tol=tol,maxIter=10,out=2)
		# @test xtt[2]==-9
		# @test all(xtt[1].==0)
		# @test length(xtt[1])==100
		# @test eltype(xtt[1])==eltype(rhs)
		
		
		x1 = qmr(A,rhs,tol=tol,maxIter=100)
		x3 = qmr(A,rhs,tol=tol,maxIter=100,x=randn(size(rhs)))
		x4 = qmr(A,rhs,tol=tol,maxIter=100,M=M3)
		
		@test norm(A*x1[1]-rhs)/norm(rhs) < tol
		@test norm(A*x3[1]-rhs)/norm(rhs) < tol
		@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-2
	end
end