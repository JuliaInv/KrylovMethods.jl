export cgls

function cgls{T1,T2}(A::SparseMatrixCSC{T1,Int},b::Array{T2,1}; kwargs...) 
    T  = promote_type(T1,T2)
    x1 = zeros(T,size(A,1))
    x2 = zeros(T,size(A,2))
    
    Af(x,flag) = (flag=='F') ? A_mul_B!(1.0,A,x,0.0,x1) : At_mul_B!(1.0,A,x,0.0,x2)
    return cgls(Af,b;kwargs...)
end

cgls(A,b::Vector;kwargs...) = cgls((x,flag) -> ((flag=='F') ? A*x : A'*x),b::Vector;kwargs...)

"""
x,flag,err,iter,resvec = cg(A,b,tol=1e-2,maxIter=100,x=[],storeInterm=false,out=0)

CGLS Conjugate gradient algorithm applied implicitly to the normal equations 

		(A'*A) x = A'*b.

Input:

  A            - function computing A*x = A(x,'F') and A'*x = A(x,'T')
  b            - right hand side vector
  tol          - error tolerance, default 1e-2
  maxIter      - maximum number of iterations, default 100
  x            - starting guess
  storeInterm  - flag for returning intermediate solutions (useful in inverse problems)
  out          - flag for output (-1: no output, 0 : only errors, 1 : final status, 2: error at each iteration)

Output:

  x       - approximate solution (interm==0) or history of approximate solutions (interm==1)
  flag    - exit flag (  0 : desired tolerance achieved,
                        -1 : maxIter reached without converging
                        -2 : Matrix A is not positive definite 
                        -9 : right hand side was zero)
  eta     - residual norm: norm(A*x-b)
  rho     - norm of current iterate: norm(x)

"""
function cgls(A::Function,b::Vector; tol::Real=1e-2,maxIter::Int=100,x::Vector=[],storeInterm::Bool=false,out::Int=0)
    
    m = length(b)
    # Initialization.
    if norm(b)==0; return 0.0,-9,0.0,0.0,[0.0]; end
    if isempty(x) || all(x.==0.0)
        r = copy(b)			# residual r = b - A*x
        s = A(r,'T')		# compute gradient g = A'*(A*x-b)
        n = length(s)
        x = zeros(n)
    else
        r = b-A(x,'F')
        s = A(r,'T')
        n = length(s)
    end
    	
    if storeInterm
        X = zeros(n,maxIter)	# allocate space for intermediates
    end
    
    p      = copy(s)
    normS0 = BLAS.dot(n,s,1,s,1)       # squared norm of gradients
    normSc = normS0
    Arn    = zeros(maxIter)
    eta    = zeros(maxIter) # norm of residuals
    rho    = zeros(maxIter) # norm or current iterate
    
    # Iterate.
    flag = -1
    if out==2
        println("=== cgls ===")
        println(@sprintf("%4s\t%8s\t%8s\t%8s","iter","|A'r|","norm(r)","norm(x)"))
    end
    
    iter = 1 # makes iter available outside the loop
    for iter=1:maxIter
        q     = A(p,'F') # compute A*g
        alpha = normSc/BLAS.dot(m,q,1,q,1)
        BLAS.axpy!(n,alpha,p,1,x,1) # faster than x    += alpha*p
        
        if storeInterm; X[:,iter] = x; end
        
        BLAS.axpy!(m,-alpha,q,1,r,1) # faster than r  -= alpha*q
        s   = A(r,'T') # compute gradient, that is A'*r
        
        normSt = BLAS.dot(n,s,1,s,1)
        if (iter>1) && (normSt <= tol)
            resize!(eta,iter-1)
            resize!(rho,iter-1)
            resize!(Arn,iter-1)
            flag = 0; break
        end
        
        beta = normSt/normSc
        
        BLAS.scal!(n,beta,p,1)
        BLAS.axpy!(n,1.0,s,1,p,1)
        # faster than p   = s + beta*p
        
        # store intermediates and report resuls
        normSc = normSt
        Arn[iter] = normSt
        eta[iter] = BLAS.nrm2(m,r,1) # faster than norm(r)
        rho[iter] = BLAS.nrm2(n,x,1) # faster than norm(x)
        if out==2;  println(@sprintf("%3d\t%1.2e\t%1.2e\t%1.2e",iter,Arn[iter],rho[iter],eta[iter]));end
    end
    
    if out>=0
        if flag==-1
            println(@sprintf("cgls iterated maxIter (=%d) times witout reaching tolerance. Returned result has residual norm  %1.2e.",
            																										maxIter,Arn[end]))
        elseif flag==0 && out>=1
            println(@sprintf("cgls relative gradient norm below desired tolerance at iteration %d. Returned result has residual norm  %1.2e.",
            																										iter,Arn[end]))
        end
    end
    
    if storeInterm
        return X[:,1:iter],flag,rho,eta,Arn
    else
        return x,flag,rho,eta,Arn
    end
end
