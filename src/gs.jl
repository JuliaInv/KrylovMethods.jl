export cgs, cgs!, mgs, mgs!

"""
KrylovMethods.cgs = KrylovMethods.cgs!(copy(V))
	
	Classical Gram Schmidt orthogonalization.

"""
cgs(V) = cgs!(copy(V))


"""
function KrylovMethods.cgs!
	
Inplace Classical Gram Schmidt orthogonalization.

Reference: page 254 in Golub and Van Loan, Matrix Computation, 4th edition.

Input:

	V::Array  - m by n matrix of full rank m<=n
	
Output:

	V::Array  - m-by-m unitary matrix
	R::Array  - m-by-n upper triangular matrix

"""
function cgs!(V::Array{T}) where {T}

	m,n    = size(V)
	R      = zeros(T,n,n)
	R[1,1] = norm(V[:,1])
	V[:,1] ./= R[1,1]
	for k=2:n
	    R[1:k-1,k] = V[:,1:k-1]'*V[:,k]
	    V[:,k]     = V[:,k]-V[:,1:k-1]*R[1:k-1,k]
	    R[k,k]     = norm(V[:,k])
	    V[:,k]   ./= R[k,k]
	end
	return V,R
end


"""
KrylovMethods.mgs = KrylovMethods.mgs!(copy(V))
	
	Modified Gram Schmidt orthogonalization.

"""
mgs(V) = mgs!(copy(V))

"""
function KrylovMethods.mgs!
	
Inplace Modified Gram Schmidt orthogonalization.

Reference: page 255 in Golub and Van Loan, Matrix Computation, 4th edition.

Input:

	V::Array  - m by n matrix of full rank m<=n
	
Output:

	V::Array  - m-by-m unitary matrix
	R::Array  - m-by-n upper triangular matrix

"""
function mgs!(V::Array{T}) where {T}
	m,n    = size(V)
	R      = zeros(T,n,n)
	for k=1:n
		R[k,k]    = norm(V[:,k])
		V[:,k]    ./= R[k,k]
		for j=k+1:n
			R[k,j] = dot(V[:,k],V[:,j])
			V[:,j] -= R[k,j]*V[:,k]
		end
	end
	return V,R
end