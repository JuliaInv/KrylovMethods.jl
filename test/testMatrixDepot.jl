using KrylovMethods
using MatrixDepot
using Base.Test

println("Tests for symmetric and positive definite A")
# debit = [ ("hilb", 5), ("cauchy",4), ("circul",50), ("invhilb",4), ("moler",7),("pascal",6),("pei",100),("tridiag",500),("lehmer",200),("poisson",20),("wathen",20)] 
debit = [ ("hilb", 5),  ("cauchy",4), ("invhilb",4), ("moler",7),("pascal",6),("pei",100),("tridiag",200),("lehmer",100),("poisson",12),("wathen",12)] 

for k=1:length(debit)
	A   = matrixdepot(debit[k][1],debit[k][2])
	rhs = randn(size(A,1))
	@printf "%s\tn=[%d,%d]\t" debit[k][1] size(A,1) size(A,2)

	xt  = A\rhs
	
	x3, = gmres(A,rhs,10,maxIter=50*length(rhs),tol=1e-16,out=0)	
	@test norm(xt-x3)/norm(xt) < 1e-3
	@printf " gmres is OK "
	x1, = cg(A,rhs,maxIter=length(rhs),tol=1e-16,out=-1)
	@test norm(xt-x1)/norm(xt) < 1e-3
	@printf "cg is OK\n"
end
	