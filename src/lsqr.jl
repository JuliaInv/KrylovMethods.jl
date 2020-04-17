export lsqr


"""
x,flag,err,iter,resvec = lsqr(A,b,;x=[],maxIter=20,atol=1e-10,btol=1e-10,condlim=1e6,out=1,doBidiag=false,
                              storeInterm::Bool=false)

Least Squares QR method (LSQR) for solving 

   min_x || A x - b ||,

where A is assumed is either over or underdetermined.

Implementation follows and comments refer to Algorithm LSQR in:

Paige, C. C., & Saunders, M. A. (1982). 
LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares. 
ACM Transactions on Mathematical Software (TOMS), 8(1), 43–71. doi:10.1145/355984.355989


Required input:

  A       - function in (flag,v,alpha,x). 
            if flag=='F' 
                 x <- A*v + alpha*x (overwrite x)
            elseif flag=='T'
                 x <- A'*v + alpha*x (overwriting x)
            end
  b       - right hand side vector

Optional input:
 
  x           - starting guess (optional)
  maxIter     - maximum number of iterations
  atol        - error tolerance for (estimated) residual norm (for compatible systems)
  btol        - error tolerance for (estimated) gradient norm (for incompatible systems)
  condlim     - for stopping based on (estimated) condition number
  out         - flag for output (-1: no output, 0: only errors, 1: final status, 2: residual norm at each iteration)
  doBidiag    - returns the bidiagonalization of A at the final iteration
  storeInterm - store and return intermediate iterates

Output:

 x       - approximate solution
 flag    - exit flag ( 0 : stopping based on atol or btol,
                      -1 : maxIter reached without converging
                      -2 : condition number became too large
                      -9 : right hand side was zero)
 his     - status at each iteration
 U,S,V   - bidiagonalization (only build if doBidiag==true)
"""
function lsqr(A::SparseMatrixCSC,b::Vector;kwargs...)
    # build function that is compatible with lsqr
    ATv = zeros(size(A,2)) # output for A'*x
    
    # Af(flag,x,a=0.0,v=ATv) = (flag=='F') ? A_mul_B!(1.0,A,x,a,v) : At_mul_B!(1.0,A,x,a,v)
    Af(flag,x,a=0.0,v=ATv) = (flag=='F') ? mul!(v,A,x,1.0,a) : mul!(v,transpose(A),x,1.0,a)

    return lsqr(Af,b;kwargs...)
    
end

function lsqr(A::Any,b::Vector;kwargs...)
    # build function that is compatible with lsqr
    Af(flag,x,a=0.0,v=0.0) = (flag=='F') ? A*x.+a*v : A'*x.+a*v

    return lsqr(Af,b;kwargs...)
    
end



function lsqr(A::Function,b::Vector;x=[],maxIter=20,atol=1e-10,btol=1e-10,condlim=1e6,out=0,doBidiag=false,storeInterm::Bool=false)
    m  = length(b)
    
    nres0  = BLAS.nrm2(m,b,1) # norm(b)
    if nres0==0; return zero(eltype(b)),-9,[0.0]; end
    
    # (1) initialize
    if !isempty(x)
        b = copy(b)
        b = A('F',-x,1.0,b)
    end
    
    nrmb  = BLAS.nrm2(m,b,1) # norm(b)
    beta  = nrmb
  
    u     = b/beta
    v     = A('T',u,0.0)# A'*u
    n     = length(v)
	x     = (isempty(x)) ? zeros(n) : x
	if storeInterm;   X = zeros(n,maxIter) end
	if (beta <= btol*nres0); return x,1; end
	alpha = BLAS.nrm2(n,v,1)
	if alpha==0; return x,2; end
    v    /= alpha
    w     = copy(v)
    
    phib  = beta
    rhob  = alpha

    
    # (2) perform iteration
    # maxIter = min(maxIter,n,m)
    nBk  = 0.0
    nDk  = 0.0
    nres = beta
    STOP = fill(false,3)
    if doBidiag
        U    = zeros(m,maxIter+1)
        U[:,1] = u
        V    = zeros(n,maxIter)
        bs   = zeros(maxIter)
        as   = zeros(maxIter)
    end
    his  = zeros(maxIter,4)
    if out>1
        @printf "iter\t|r_k|\t\t|A'*r_k|\t|A|\t\tcond(A)\t\tSTOP\n"
    end
    flag = -1
    iter = 1
    while iter<=maxIter
        # (3) continue the bidiagonalization
        u     = A('F',v,-alpha,u) # A*v - alpha*u
        beta  = BLAS.nrm2(m,u,1)
        BLAS.scal!(m,1/beta,u,1) # faster than: u    /= beta
        if doBidiag
            as[iter] = alpha
            bs[iter] = beta
            U[:,iter+1] = u
            V[:,iter]   = v
        end
        nBk  += alpha^2+beta^2 
        v     = A('T',u,-beta,v) # A'*u - beta*v
        alpha = BLAS.nrm2(n,v,1)
        BLAS.scal!(n,1/alpha,v,1) # faster than: v    /= alpha
        
        # (4) construct and apply next orthogonal transformation
        rho   = sqrt(rhob^2+beta^2)
        c     = rhob/rho
        s     = beta/rho
        theta = s*alpha
        rhob  = -c*alpha
        phi   = c*phib
        phib *= s
        nDk  += dot(w,w)/(rho*rho)
        
        # (5) update x,w
        BLAS.axpy!(n,phi/rho,w,1,x,1) # x += (phi/rho)*w
        if storeInterm; X[:,iter] = x; end
		
        # the following two lines are equivalent to w  = v - (theta/rho)*w 
        w = BLAS.scal!(n,-theta/rho,w,1)
        w = BLAS.axpy!(n,1.0,v,1,w,1)
       
        # (6) check for stopping
        nrmA = sqrt(nBk)
        nrmG = abs(phib*alpha*c)
        conA = sqrt(nBk*nDk)
        nres *= s
        his[iter,:] = [nres nrmG nrmA conA]
        
        STOP[1] = nres <= btol*nres0+ atol*nrmA*norm(x)        
        STOP[2] = (nrmG/(nrmA*nres)) <= atol
        STOP[3] = conA > condlim
        
        if out>1
            @printf "%03d\t%1.3e\t%1.3e\t%1.3e\t%1.3e\t[%d,%d,%d]\n" iter nres nrmG nrmA conA 1.0*STOP[1] 1.0*STOP[2] 1.0*STOP[3]
        end
        iter += 1
        
        if any(STOP)
            break; 
        end
    end

    if STOP[1]; flag = 1; end
    if STOP[2] && !STOP[1]; flag = 2; end
    if STOP[3]; flag = -2; end

    if out>=0
        if flag==-1
            println(@sprintf("lsqr iterated maxIter (=%d) times without converging to desired tolerance. Estimated residual norm is %1.3e and estimated norm of gradient is %1.3e.",iter-1,his[iter-1,1],his[iter-1,2]))
        elseif flag==-2
            println(@sprintf("lsqr stopped at iteration %d because the matrix seems to be too ill-conditioned. Estimated condition number is %1.2e.",iter-1,his[iter-1,4]))
        elseif flag==1 && out>=1
            println(@sprintf("lsqr achieved desired tolerance at iteration %d. The system should be compatible. Residual norm is %1.2e.",iter-1,his[iter-1,1]))
        elseif flag==2 && out>=1
            println(@sprintf("lsqr achieved desired tolerance at iteration %d. The system should be incompatible. Gradient norm is %1.2e.",iter-1,his[iter-1,1]))
        end
    end

    x = storeInterm ? X[:,1:iter-1] : x;
    
    if doBidiag
        U = U[:,1:iter]
        V = V[:,1:iter-1]
		I,J,V = SparseArrays.spdiagm_internal(-1=>bs[1:iter-1],0=>as[1:iter-1])
		B = sparse(I,J,V,iter,iter-1)
        # B = spdiagm((bs[1:iter-1],as[1:iter-1]),-1:0,iter,iter-1)
        return x, flag,his[1:iter-1,:],U,B,V
    else
        return x, flag,his[1:iter-1,:]
    end
end