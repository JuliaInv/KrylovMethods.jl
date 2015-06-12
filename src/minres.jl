export minres

# x,flag,err,iter,resvec = minres(A,b,;x=[],maxIter=20,tol=1e-10,out=1)
#
# Minimal Residual Method (MINRES) for solving
#
#    A x = b,
#
# where A is symmetric and possibly indefinite.
#
# Implementation follows page 26 of
#
# Choi, S.-C. T. (2006). 
# Iterative Methods for Singular Linear Equations and Least-squares Problems. 
# Phd thesis, Stanford University.
# 
# See also:
#
# Paige, C. C., & Saunders, M. A. (1975). 
# Solution of sparse indefinite systems of linear equations. 
# SIAM Journal on Numerical Analysis.
# 
#
# Required input:
#
#   A       - function that performs matrix-vector product, e.g., x -> A*x
#   b       - right hand side vector
# 
# Optional input:
#  
#   x        - starting guess (optional)
#   maxIter  - maximum number of iterations
#   btol     - error tolerance for Lanczos step
#   rtol     - error tolerance for estimated relative residual
#   gtol     - error tolerance for estimated gradient norm
#   condlim  - limit on condition number esitmate of A
#   out      - flag for output (-1: no output, 0: only errors, 1: final status, 2: residual norm at each iteration)
#
# Output:
#
#  x       - approximate solution
#  flag    - exit flag (0: converged, -1: maxIter, -2: Lanczos, -3: condition)
#  relres  - estimated relative residual at final iteration
#  nrmG    - estimated gradient norm at final iteration
#  nrmA    - estimated Frobenius norm of A
#  conA    - condition number estimate
#  phi     - relres for each iteration

function minres{T1,T2}(A::SparseMatrixCSC{T1,Int},b::Array{T2,1}; kwargs...) 
	x = zeros(promote_type(T1,T2),size(A,2)) # pre-allocate
	return minres(v -> At_mul_B!(1.0,A,v,0.0,x),b;kwargs...) # multiply with transpose of A for efficiency
end

minres(A,b;kwargs...) = minres(x -> A*x,b::Vector;kwargs...)


function minres(A::Function,b;x=[],sigma=0.0,btol=1e-10,rtol=1e-10,gtol=1e-10,condlim=1e10,maxIter=10,out=1)
    
    n      = length(b)
    if all(b.==0); return zeros(eltype(b),n); end
    
    if !isempty(x)
		b -= A(x)
	else
    	x = zeros(eltype(b),n)
	end
	
    nres = norm(b)
    
    # initialize scalars (or vectors of size maxIter)
    alpha    = zeros(maxIter+1)
    beta     = zeros(maxIter+2)
    beta[2]  = nres
    gamma    = zeros(maxIter+2)
    gamma[1] = nres
    epsil    = zeros(maxIter+2)
    delta    = zeros(maxIter+2)
    delta[2] = 0.0
    phi      = zeros(maxIter+1)
    phi[1]   = nres
    psi      = zeros(maxIter+1)
    tau      = zeros(maxIter+1)
    tau[1]   = beta[2]
    
    conA     =  1.0
    gammaMin =  Inf
    nrmG     =  0.0
    c        = -1.0
    s        =  0.0

    # allocate vectors of size n
    vk   = b/beta[2]
    vkm1 = zeros(eltype(b),n)
    dkm1 = zeros(eltype(b),n)
    dkm2 = zeros(eltype(b),n)
    
    if out>1
		println("=== minres ===")
        @printf "iter\t|r_k|\t\t|A*r_k|\t\t|A|\t\t\tcond(A)\n"
    end
    flag = -1
 
    k = 2
    nrmA = alpha[1]*beta[2]
    for k=2:maxIter+1
        alpha[k],beta[k+1],vk,vkm1 = LanczosStep!(A,vk,vkm1,beta[k],sigma=sigma,tol=btol)
        
        if beta[k+1] < btol
            flag = -2
            break
        end
        
        deltab     = c*delta[k] + s*alpha[k]
        gamma[k]   = s*delta[k] - c*alpha[k]
        epsil[k+1] = s*beta[k+1]
        delta[k+1] =-c*beta[k+1]
        
        c,s,gammab = symOrtho(gamma[k],beta[k+1])
        
        tau[k]   = c*phi[k-1]
        phi[k]   = s*phi[k-1]
        nrmG     = phi[k]*sqrt(gamma[k+1]^2+delta[k+1]^2)
            
        psi[k-1] = phi[k-1] * sqrt(gamma[k]^2+delta[k+1]^2)
        
        nrmA = max(nrmA,sqrt(beta[k]^2 + alpha[k]^2 + beta[k+1]^2))
        
        if gammab != 0
            tt = copy(dkm1)
            dkm1  = vkm1 - deltab*dkm1 
            dkm1 -= epsil[k]*dkm2
            dkm1 /= gammab
            dkm2  = tt
            x       += tau[k]*dkm1
            gammaMin = min(gammaMin,gammab)
            conA     = nrmA/gammaMin
        end
        if out>1
            @printf "%d\t%1.2e\t%1.2e\t%1.2e\t\t%1.2e\n" k-1 phi[k]/nres nrmG nrmA conA 
        end
        if (phi[k]/nres) < rtol && (nrmG < gtol)
            flag = 0
            break
        end

        if conA > condlim
            flag = -3
            break
        end
    end
    if out>=0
        if flag==-1
            println(@sprintf("minres iterated maxIter (=%d) times but 
                reached only residual norm %1.2e instead of tol=%1.2e.",
            maxIter,phi[k]/nres,rtol))
        end
        if flag==-2
            println(@sprintf("minres stopped because beta (=%1.2e) became
            smaller than the tolerance (btol=%1.2e) at iteration %d.",
            beta[k+1],btol,k-1))
        end
        if flag==-3
            println(@sprintf("The estimated condition number (=%1.2e) in minres 
            is bigger than the tolerance (=%1.2e) at iteration %d.",
            conA,condlim,k-1))
        end
        if (out>0) && flag==0
            println(@sprintf("minres converged at iteration %d. 
            |A r_k|=%1.2e and |r_k|=%1.2e.",k-1,nrmG,phi[k]/nres))
        end
    end
    return x,flag,phi[k]/nres,nrmG, nrmA,phi
end
