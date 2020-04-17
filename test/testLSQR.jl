
@testset "LSQR" begin

	@testset "small test" begin
		A   = [1 1; 0 -1; 1 0.]
		r1  = A[:,1]   # compatible
		r2  = randn(3) # incompatible
		r3  = [1;1;-1.] # in the nullpace of A'

		x1 = lsqr(A,r1,out=2,atol=1e-8,doBidiag=true,storeInterm=true)
		@test norm(x1[1][:,end]-[1;0])<1e-10
		@test x1[2]==1
		x2 = lsqr(A,r2,out=1,atol=1e-8)
		@test x2[2]==2
		x3 = lsqr(A,r3,out=2)
		@test all(x3[1].==0)
	end


	@testset "sparse random" begin
		A   = sprandn(100,10,.2)
		rhs = randn(100)
		xgt = Matrix(A)\rhs
		Af = (flag,x,a=0.0,v=0.0) -> ((flag=='F') ? A*x.+a*v : A'*x.+a*v)

		x1  = lsqr(LinearOperator(A),rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10,out=1)
		x2  = lsqr(A,rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10)
		x3  = lsqr(Af,rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10)

		@test norm(xgt-x1[1])/norm(xgt) < 1e-6
		@test norm(xgt-x2[1])/norm(xgt) < 1e-6
		@test norm(xgt-x3[1])/norm(xgt) < 1e-6
		@test norm(x1[1]-x2[1])/norm(x1[1]) < 1e-12
		@test norm(x2[1]-x3[1])/norm(x3[1]) < 1e-12
	end
	# @testset "hilbert matrix" begin
	# 	A   = matrixdepot("hilb",3,3)
	# 	rhs = [1;0;0.]
	# 	xgt = A\rhs
	#
	# 	x1  = lsqr(LinearOperator(A),rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10)
	# 	x2  = lsqr(A,rhs,atol=1e-10,btol=1e-10,condlim=1e5,maxIter=10)
	#
	# 	@test norm(xgt-x1[1])/norm(xgt) < 1e-6
	# 	@test norm(xgt-x2[1])/norm(xgt) < 1e-6
	# 	@test norm(x1[1]-x2[1])/norm(x1[1]) < 1e-12
	# 	conA = norm(vec(A))*norm(vec(pinv(A)))
	# 	@test norm(conA-x1[3][end,end])/conA < 1e-12
	# 	@test norm(conA-x2[3][end,end])/conA < 1e-12
	# end

end
