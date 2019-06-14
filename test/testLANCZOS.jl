@testset "lanczos" begin
	@testset "givens" begin
		symOrth = KrylovMethods.symOrtho

		v     = ([0;randn()],[randn();0],randn(2))
		for k=1:3
			println("Givens rotation for v=[$(v[k][1]),$(v[k][2])]")
			c,s,r = symOrth(v[k][1],v[k][2])
			@test abs(c.^2 + s.^2 - 1) < 1e-14
			Q = [c s; -s c]
			w = [norm(v[k]);0]
			@test norm(Q*v[k]-w) < 1e-13
		end
	end
	@testset "bidiag" begin
		println("=== Testing Lanczos Bidiagonalization ===")
		A = sprandn(10,10,.4) + I
		p = randn(10)
		k = 9

		U,B,V = lanczosBidiag(A,p,k)

		@test norm(Matrix(U'*A*V-B))/norm(Matrix(A)) < 1e-15
	end


	@testset "tridiag" begin
		# test stopping for low-rank matrix
		A = randn(100,10)
		H = A*A'
		T,V = lanczosTridiag(H,ones(100),100,doReorth=true)
		@test size(T,1)==11
		@test size(V,2)==11

		# A   = matrixdepot("rosser",4)
		# Aff  = x -> A*x
		# T1,V1 = lanczosTridiag(A,ones(size(A,1)),size(A,1),tol=1e-10,doReorth=true)
		# T2,V2 = lanczosTridiag(Aff,ones(size(A,1)),size(A,1),tol=1e-10,doReorth=false)
		# sT1  = svd(Matrix(T1)).S
		# sT2  = svd(Matrix(T2)).S
		# sA   = svd(Matrix(A)).S
		#
		# @test abs(sT1[1]-sA[1])/sA[1] < 1e-4
		# @test abs(sT2[1]-sA[1])/sA[1] < 1e-4
		# B = V1'*V1
		# @test norm(B - Diagonal(diag(B))) < 1e-10
		# @test norm(diag(B)- ones(size(B,1))) < 1e-10
	end
end
