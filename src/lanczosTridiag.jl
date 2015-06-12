export lanczosTridiag

# T,V = lanczosTridiag(A,b,k;tol,doReorth)
#
# Lanczos method for getting a factorization of
#
#    A = Vk Tk Vk'
#
# where A is a real symmetric n by n matrix, Tk is a tridiagonal k by k matrix and the columns of 
# the n by k matrix Vk are orthogonal.
#
# Implementation follows:
#
# Paige, C. C. (1972). 
# Computational variants of the Lanczos method for the eigenproblem. 
# IMA Journal of Applied Mathematics. 
#
# Required input:
#
#   A       - function computing A*x, e.g., x -> A*x
#   b       - right hand side vector
#   k       - dimension of Krylov subspace
# 
# Optional input:
#  
#   tol      - stopping tolerance
#   doReorth - (default=false) set to true to perform full reorthogonalization
#
# Output:
#
#  Tk    - sparse tridiagonal matrix
#  Vk    - basis vectors

function lanczosTridiag{T1,T2}(A::SparseMatrixCSC{T1,Int},b::Array{T2,1},k; kwargs...) 
	x = zeros(promote_type(T1,T2),size(A,2)) # pre-allocate
	return lanczosTridiag(v -> At_mul_B!(1.0,A,v,0.0,x),b,k;kwargs...) # multiply with transpose of A for efficiency
end


lanczosTridiag(A,b,k;kwargs...) = lanczosTridiag(x -> A*x,b,k;kwargs...)


function lanczosTridiag(A::Function,b::Vector,k::Int;tol=1e-5,doReorth::Bool=false)

n   = length(b)
x   = zeros(n)

# pre-allocate space for tridiagonalization and basis
beta  = zeros(k)
alpha = zeros(k)
V     = zeros(n,k)


beta[1] = norm(b)
V[:,1]  = copy(b)/beta[1]
u       = A(V[:,1])

j = 1 # brings j to scope of function
for j=1:k-1
    alpha[j] = dot(V[:,j],u)
    u        = u - alpha[j]*V[:,j]
    if doReorth # full re-orthogonalization
        for i=1:j
            u -= V[:,i]*dot(V[:,i],u)
        end
    end
    gamma     = norm(u)
    V[:,j+1]  = u/gamma
    beta[j+1] = gamma
    if beta[j+1]<tol
        break
    end
    u = A(V[:,j+1]) - beta[j+1]*V[:,j]
end

T = spdiagm((beta[2:j],alpha[1:j],beta[2:j]),-1:1,j,j)
return T,V[:,1:j]
end
