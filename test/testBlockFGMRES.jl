
println("*******************************************************")
@testset "blockFGMRES" begin
	@testset "real matrix" begin
		n = 50;
		A  = sprandn(n,n,.1) + SparseMatrixCSC(10.0I, n, n)
		m  = 2; 
		D  = Vector(diag(A))
		M2 = x -> D.\x
		rhs = randn(n,m)
		tol = 1e-6;
		
		# test printing and behaviour for early stopping
		xtt = blockFGMRES(A,rhs ,3,tol=1e-12,maxIter=3,out=2)
		@test xtt[2]==-1
		
		# test behaviour for zero rhs
		xtt = blockFGMRES(A,0*rhs,3,tol=tol,maxIter=10,out=2)
		@test xtt[2]==-9
		@test all(xtt[1].==0.0)
		@test length(xtt[1])==m*n
		@test eltype(xtt[1])==eltype(rhs)

		println("-----------------------------------")
		x1 = blockFGMRES(A,rhs ,5,tol=tol,maxIter=100,out=1)
		x3 = blockFGMRES(A,rhs,5,tol=tol,maxIter=100,X=randn(size(rhs)),flexible=true)
		x4 = blockFGMRES(A,rhs,5,tol=tol,maxIter=100,M=M2,flexible = true)
		
		@test norm(A*x1[1]-rhs)/norm(rhs) < tol
		@test norm(A*x3[1]-rhs)/norm(rhs) < tol
		@test norm(A*x4[1]-rhs)/norm(rhs) < tol
		@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-5
	end
	println("*****************************************************************************")
	@testset "complex matrix" begin
		A  = sprandn(100,100,.1) + SparseMatrixCSC(10.0I, 100, 100) + 1im*(sprandn(100,100,.1) + SparseMatrixCSC(10.0I, 100, 100) )
		m  = 3; 
		D  = Vector(diag(A))
		M3 = x -> x./D
		rhs = complex(randn(100,m))
		tol = 1e-6;
		
		# test behaviour for zero rhs
		xtt = blockFGMRES(A,0.0*rhs,5,tol=tol,maxIter=10,out=2)
		@test xtt[2]==-9
		@test all(xtt[1].==0)
		@test length(xtt[1])==100*m
		@test eltype(xtt[1])==eltype(rhs)
		
		
		x1 = blockFGMRES(A,rhs ,5,tol=tol,maxIter=100)
		x3 = blockFGMRES(A,rhs,5,tol=tol,maxIter=100,X=randn(size(rhs)),flexible = true)
		x4 = blockFGMRES(A,rhs,5,tol=tol,maxIter=100,M=M3,flexible = true)
		
		@test norm(A*x1[1]-rhs)/norm(rhs) < tol
		@test norm(A*x3[1]-rhs)/norm(rhs) < tol
		@test norm(x3[1]-x1[1])/norm(x1[1]) < 1e-5
	end
end