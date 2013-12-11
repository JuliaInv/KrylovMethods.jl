export lanczosBidiag

function lanczosBidiag(A,p::Vector,k::Int)
# U, B, V =  lanczosBidiag(A,p::Vector,k::Int)
#
# Lanczos bidiagonalization of matrix A
#
# Input:
#
#   A       - matrix or function computing A*x = A(x,'F') and A'*x = A(x,'T')
#   p       - starting vector
#   k       - dimension of subspace
#
# Output:
#
#   U,B,V   - solution

Af(x)  =  isa(A,Function) ? A(x,'F') : A*x
ATf(x) =  isa(A,Function) ? A(x,'T') : A'*x

# Initialization.
m      = length(p)
beta   = norm(p)
u      = p/beta

ATu = ATf(u)

n   = length(ATu)
v   = zeros(n)

# alocate space for Lancszos vectors
U = zeros(m,k+1)
V = zeros(n,k)
B = zeros(k,2)
# Prepare for Lanczos iteration.
U[:,1] = u

# Perform Lanczos bidiagonalization with reorthogonalization.
for i=1:k
    if i>1;
        ATu = ATf(u)
    end
    r = ATu - beta*v

    # reorthogonalization of V
    for j=1:i-1
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
    u = p/beta
    B[i,1] = beta
    U[:,i+1] = u
end

B = spdiagm((B[:,1],B[:,2]),[-1,0],k+1,k)

return U, B, V

end
