export cgs, mgs


"""
function KrylovMethods.cgs
	
Classical Gram Schmidt orthogonalization.

Reference: page 254 in Golub and Van Loan, Matrix Computation, 4th edition.

Input:

	V::Array  - m by n matrix of full rank m<=n
	
Output:

	Q::Array  - m-by-m unitary matrix
	R::Array  - m-by-n upper triangular matrix

"""
function cgs{T}(V::Array{T})

	m,n    = size(V)
	R      = zeros(T,n,n)
	Q      = zeros(T,m,n)
	R[1,1] = norm(V[:,1])
	Q[:,1] = V[:,1]./R[1,1]
	for k=2:n
	    R[1:k-1,k] = Q[:,1:k-1]'*V[:,k]
	    Q[:,k]     = V[:,k]-Q[:,1:k-1]*R[1:k-1,k]
	    R[k,k]     = norm(Q[:,k])
	    Q[:,k]     = Q[:,k]./R[k,k]
	end
	return Q,R
end

"""
function KrylovMethods.mgs
	
Modified Gram Schmidt orthogonalization.

Reference: page 255 in Golub and Van Loan, Matrix Computation, 4th edition.

Input:

	V::Array  - m by n matrix of full rank m<=n
	
Output:

	Q::Array  - m-by-m unitary matrix
	R::Array  - m-by-n upper triangular matrix

"""
function mgs{T}(V::Array{T})
	V      = copy(V)
	m,n    = size(V)
	R      = zeros(T,n,n)
	Q      = zeros(T,m,n)
	for k=1:n
		R[k,k]     = norm(V[:,k])
		Q[:,k]     = V[:,k]/R[k,k]
		for j=k+1:n
			R[k,j] = dot(Q[:,k],V[:,j])
			V[:,j] -= R[k,j]*Q[:,k]
		end
	end
	return Q,R
end