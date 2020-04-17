export minres

"""
 x,flag,err,iter,resvec = minres(A,b,;x=[],maxIter=20,tol=1e-10,out=1)

 Minimal Residual Method (MINRES) for solving

    A x = b,

 where A is symmetric and possibly indefinite.

 Implementation follows page 26 of

 Choi, S.-C. T. (2006). 
 Iterative Methods for Singular Linear Equations and Least-squares Problems. 
 Phd thesis, Stanford University.
 
 See also:

 Paige, C. C., & Saunders, M. A. (1975). 
 Solution of sparse indefinite systems of linear equations. 
 SIAM Journal on Numerical Analysis.
 

 Required input:

   A       - function that performs matrix-vector product, e.g., x -> A*x
   b       - right hand side vector
 
 Optional input:
  
   x        - starting guess (optional)
   maxIter  - maximum number of iterations
   btol     - error tolerance for Lanczos step
   rtol     - error tolerance for estimated relative residual
   gtol     - error tolerance for estimated gradient norm
   condlim  - limit on condition number esitmate of A
   out      - flag for output (-1: no output, 0: only errors, 1: final status, 2: residual norm at each iteration)

 Output:

  x       - approximate solution
  flag    - exit flag (0: converged, -1: maxIter, -2: Lanczos, -3: condition)
  relres  - estimated relative residual at final iteration
  nrmG    - estimated gradient norm at final iteration
  nrmA    - estimated Frobenius norm of A
  conA    - condition number estimate
  phi     - relres for each iteration
"""
function minres(A::SparseMatrixCSC{T1,Int},b::Array{T2,1}; kwargs...) where {T1,T2}
	x = zeros(promote_type(T1,T2),size(A,2)) # pre-allocate
	return minres(v -> mul!(x,transpose(A),v,1.0,0.0),b;kwargs...) # multiply with transpose of A for efficiency
end

minres(A,b::Vector;kwargs...) = minres(x -> A*x,b::Vector;kwargs...)


function minres(A::Function,b::Vector;x=[],sigma=0.0,btol=1e-10,rtol=1e-10,gtol=1e-10,condlim=1e10,maxIter=10,out=1,storeInterm::Bool=false)
    
    n      = length(b)
    nres   = norm(b)
	if nres==0; return zeros(eltype(b),n),-9,0.0,0.0,0.0,[0.0]; end
    
    if !isempty(x)
		b -= A(x)
	else
    	x = zeros(eltype(b),n)
	end
	if storeInterm
        X = zeros(eltype(b),n,maxIter)	# allocate space for intermediates
    end
    
    
    # initialize scalars (or vectors of size maxIter)
    alpha    = zeros(maxIter+1)
    beta     = zeros(maxIter+2)
    beta[2]  = norm(b)
    gamma    = zeros(maxIter+2)
    gamma[1] = beta[2]
    epsil    = zeros(maxIter+2)
    delta    = zeros(maxIter+2)
    delta[2] = 0.0
    phi      = zeros(maxIter+1)
    phi[1]   = beta[2]
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
    tt   = zeros(eltype(b),n)
    
    if out>1
		println("=== minres ===")
        @printf "iter\t|r_k|\t\t|A*r_k|\t\t|A|\t\t\tcond(A)\n"
    end
    flag = -1
 
    nrmA = alpha[1]*beta[2]
	k = 2
    while k <= maxIter+1
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
            # the following line is equivalent to: tt = copy(dkm1)
            tt = BLAS.blascopy!(n,dkm1,1,tt,1)
            # the following two lines are equivalent to dkm1 = vkm1 - deltab*dkm1
            dkm1 = BLAS.scal!(n,-deltab,dkm1,1)
            dkm1 = BLAS.axpy!(n,1.0,vkm1,1,dkm1,1)
            dkm1 = BLAS.axpy!(n,-epsil[k],dkm2,1,dkm1,1)
            dkm1 = BLAS.scal!(n,1.0/gammab,dkm1,1)
            dkm2 = BLAS.blascopy!(n,tt,1,dkm2,1)
            
            # the following line is equivalent to x    += tau[k]*dkm1
            x    = BLAS.axpy!(n,tau[k],dkm1,1,x,1)
            gammaMin = min(gammaMin,gammab)
            conA     = nrmA/gammaMin
        end
        if storeInterm; X[:,k-1] = x; end
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
		k += 1
    end
	k = min(k,maxIter+1)
    if out>=0
        if flag==-1
            println(@sprintf("minres iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",
            maxIter,phi[k]/nres,rtol))
        end
        if flag==-2
            println(@sprintf("minres stopped because beta (=%1.2e) became smaller than the tolerance (btol=%1.2e) at iteration %d.",
            beta[k+1],btol,k-1))
        end
        if flag==-3
            println(@sprintf("The estimated condition number (=%1.2e) in minres is bigger than the tolerance (=%1.2e) at iteration %d.",
            conA,condlim,k-1))
        end
        if (out>0) && flag==0
            println(@sprintf("minres converged at iteration %d. |A r_k|=%1.2e and |r_k|=%1.2e.",k-1,nrmG,phi[k]/nres))
        end
    end

    x = (storeInterm) ? X[:,1:k-1] : x
    return x,flag,phi[k]/nres,nrmG, nrmA,phi
end
