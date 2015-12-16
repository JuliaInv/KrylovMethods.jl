export lanczosBidiag

function lanczosBidiag{T1,T2}(A::SparseMatrixCSC{T1,Int},b::Array{T2,1}, k::Int) 
	T  = promote_type(T1,T2)
	x1 = zeros(T,size(A,1))
	x2 = zeros(T,size(A,2))
	
	Af(x,flag) = (flag=='F') ? A_mul_B!(1.0,A,x,0.0,x1) : At_mul_B!(1.0,A,x,0.0,x2)
	return lanczosBidiag(Af,b,k)
end


lanczosBidiag(A,b,k) = lanczosBidiag((x,flag) -> ((flag=='F') ? A*x : A'*x),b,k)

"""
U, B, V =  lanczosBidiag(A,p::Vector,k::Int)

Lanczos bidiagonalization of matrix A.

Input:

  A       - function computing A*x = A(x,'F') and A'*x = A(x,'T')
  p       - starting vector
  k       - dimension of subspace

Output:

  U,B,V   - Lanczos vectors
"""
function lanczosBidiag(A::Function,p::Vector,k::Int)
	
	m      = length(p)
	beta   = norm(p)
	u      = p/beta
	ATu    = A(u,'T')
	
	n      = length(ATu)
	v      = zeros(n)
	
	# alocate space for Lanczos vectors
	U = zeros(m,k+1)
	V = zeros(n,k)
	B = zeros(k,2)
	# Prepare for Lanczos iteration.
	U[:,1] = u
	
	for i=1:k # perform Lanczos bidiagonalization with reorthogonalization.
		if i>1;
			ATu = A(u,'T')
		end
		r = ATu - beta*v
		
		for j=1:i-1 # reorthogonalization of V
			r -= dot(V[:,j],r)*V[:,j]
		end
		
		alpha = norm(r)
		v = r/alpha
		B[i,2] = alpha
		V[:,i] = v
		
		Av = A(v,'F')
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
