function getDivGrad(n1,n2,n3)
	# D = getDivGrad(n1,n2,n3)
	# builds 3D divergence operator
	D1 = kron(sparse(1.0I,n3*n2,n3*n2),ddx(n1))
	D2 = kron(sparse(1.0I,n3,n3),kron(ddx(n2),sparse(1.0I,n1,n1)))
	D3 = kron(ddx(n3),sparse(1.0I,n2*n1,n2*n1))
	
	Div = [D1 D2 D3]
	return Div*Div'
end

function ddx(n)
# generate 1D finite difference on staggered grid
I,J,V = SparseArrays.spdiagm_internal(0=>-ones(n),1=>ones(n) )
return sparse(I,J,V,n,n+1)
	# return d = spdiags(ones(n)*[-1 1],[0,1],n,n+1)
end

function spdiags(B,d,m,n)
# A = spdiags(B,d,m,n)
# creates a sparse matrix from its diagonals
	d = d[:]
	p = length(d)
	len = zeros(p+1,1)
	for k = 1:p
		len[k+1] = round.(Int,len[k]+length(max(1,1-d[k]):min(m,n-d[k])))
	end
	a = zeros(round.(Int,len[p+1]),3)
	for k = 1:p
		# Append new d[k]-th diagonal to compact form
		i = max(1,1-d[k]):min(m,n-d[k])
		a[(round.(Int,len[k])+1):round.(Int,len[k+1]),:] = [i i+d[k] B[i+(m>=n)*d[k],k]]
	end
	
	A = sparse(round.(Int,a[:,1]),round.(Int,a[:,2]),a[:,3],m,n)
	return A
end
