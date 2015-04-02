export lanczosBidiag

function lanczosBidiag{T}(A::SparseMatrixCSC{T,Int64},p::Vector,k::Int) 
	x1 = zeros(T,size(A,1))
	x2 = zeros(T,size(A,2))
	
	Alinop = LinearOperator(size(A,1),size(A,2),T,false,false,
							v -> A_mul_B!(1.0,A,v,0.0,x1),nothing,
							v -> At_mul_B!(1.0,A,v,0.0,x2))
	return lanczosBidiag(Alinop,p,k)
end

lanczosBidiag(A::Array,p::Vector,k::Int) = lanczosBidiag(LinearOperator(A),p,k)




function lanczosBidiag(A::LinearOperator,p::Vector,k::Int)
# U, B, V =  lanczosBidiag(A,p::Vector,k::Int)
#
# Lanczos bidiagonalization of matrix A.
#
# Input:
#
#	A       - matrix or function computing A*x = A(x,'F') and A'*x = A(x,'T')
#	p       - starting vector
#	k       - dimension of subspace
#
# Output:
#
#	U,B,V   - Lanczos vectors

	Af  =  isa(A,Function) ? x->A(x,'F') : x->A*x
	ATf =  isa(A,Function) ? x->A(x,'T') : x->A'*x
	
	m      = length(p)
	beta   = norm(p)
	u      = p/beta
	ATu    = ATf(u)
	
	n      = length(ATu)
	v      = zeros(n)
	
	# alocate space for Lancsos vectors
	U = zeros(m,k+1)
	V = zeros(n,k)
	B = zeros(k,2)
	# Prepare for Lanczos iteration.
	U[:,1] = u
	
	for i=1:k # perform Lanczos bidiagonalization with reorthogonalization.
		if i>1;
			ATu = ATf(u)
		end
		r = ATu - beta*v
		
		for j=1:i-1 # reorthogonalization of V
			r -= dot(V[:,j],r)*V[:,j]
		end
		
		alpha = norm(r)
		v = r/alpha
		B[i,2] = alpha
		V[:,i] = v
		
		Av = Af(v)
		p = Av - alpha*u
		
		for j=1:i; p = p - dot(U[:,j],p)*U[:,j]; end
		beta = norm(p)
		u    = p/beta
		
		B[i,1]   = beta
		U[:,i+1] = u
	end
	
	B = spdiagm((B[:,1],B[:,2]),[-1,0],k+1,k)
	return U, B, V
end
