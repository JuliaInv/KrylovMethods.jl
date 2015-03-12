function getDivGrad(n1,n2,n3)
	# D = getDivGrad(n1,n2,n3)
	# builds 3D divergence operator
	D1 = kron(speye(n3),kron(speye(n2),ddx(n1)))
	D2 = kron(speye(n3),kron(ddx(n2),speye(n1)))
	D3 = kron(ddx(n3),kron(speye(n2),speye(n1)))
	
	Div = [D1 D2 D3]
	return Div*Div'
end

function ddx(n)
# generate 1D finite difference on staggered grid
	return d = spdiags(ones(n)*[-1 1],[0,1],n,n+1)
end

function spdiags(B,d,m,n)
# A = spdiags(B,d,m,n)
# creates a sparse matrix from its diagonals
	d = d[:]
	p = length(d)
	len = zeros(p+1,1)
	for k = 1:p
		len[k+1] = round(Int64,len[k]+length(max(1,1-d[k]):min(m,n-d[k])))
	end
	a = zeros(round(Int64,len[p+1]),3)
	for k = 1:p
		# Append new d[k]-th diagonal to compact form
		i = max(1,1-d[k]):min(m,n-d[k])
		a[(round(Int64,len[k])+1):round(Int64,len[k+1]),:] = [i i+d[k] B[i+(m>=n)*d[k],k]]
	end
	
	A = sparse(round(Int64,a[:,1]),round(Int64,a[:,2]),a[:,3],m,n)
	return A
end
