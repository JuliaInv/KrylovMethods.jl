export gmres

function gmres(A::SparseMatrixCSC{T1,Int},b::Array{T2,1},restrt::Int; kwargs...) where {T1,T2}
	Ax = zeros(promote_type(T1,T2),size(A,1))
    Af(flag,x,a=0.0,v=ATv) = (flag=='F') ? mul!(v,A,x,1.0,a) : mul!(v,transpose(A),x,1.0,a)
	
	# return gmres(x -> A_mul_B!(1.0,A,x,0.0,Ax),b,restrt;kwargs...)
	return gmres(x -> mul!(Ax,A,x,1.0,0.0),b,restrt;kwargs...)
end

gmres(A,b::Vector,restrt;kwargs...) = gmres(x -> A*x ,b,restrt;kwargs...)

"""
x,flag,err,iter,resvec = gmres(A,b,restrt,tol=1e-2,maxIter=100,M=1,x=[],out=0)

Generalized Minimal residual ( GMRESm ) method with restarts applied to A*x = b.

Input:

  A       - function computing A*x 
  b       - right hand side vector
  restrt  - number of iterations between restarts
  tol     - error tolerance
  maxIter - maximum number of iterations
  M       - preconditioner, function computing M\\x
  x       - starting guess
  out     - flag for output (0 : only errors, 1 : final status, 2: error at each iteration)

Output:

  x       - approximate solution
  flag    - exit flag (  0 : desired tolerance achieved,
                        -1 : maxIter reached without converging
                        -9 : right hand side was zero)
  err     - norm of relative residual, i.e., norm(A*x-b)/norm(b)
  iter    - number of iterations
  resvec  - norm of relative residual at each iteration
"""
function gmres(A::Function,b::Vector,restrt::Int; tol::Real=1e-2,maxIter::Int=100,M::Function=identity,x::Vector=[],out::Int=0,storeInterm::Bool=false)
    
    # initialization
    n  = length(b)
	if norm(b)==0; return zeros(eltype(b),n),-9,0.0,0,[0.0]; end
    if isempty(x)
        x = zeros(n)
        r = M(b)
    else
        r = M(b-A(x))
    end
	if storeInterm
        X = zeros(n,maxIter)	# allocate space for intermediates
    end
    
    bnrm2 = norm(b)
    if bnrm2 == 0.0; bnrm2 = 1.0; end
    
    err = norm( r ) / bnrm2
    if err < tol; return x, err; end
    
    restrt = min(restrt,n-1)
    V     = zeros(n,restrt+1)
    H     = zeros(restrt+1,restrt)  
    cs    = zeros(restrt)  
    sn    = zeros(restrt)
    e1    = zeros(n)
    e1[1] = 1.0
    
    if eltype(b) <: Complex || eltype(r) <: Complex || eltype(A) <: Complex
        e1 = complex(e1)
        b  = complex(b)
        r  = complex(r)
        x  = complex(x)
        V  = complex(V)
        H  = complex(H)
        cs = complex(cs)
        sn = complex(sn)
    end
    
    resvec = zeros((1+restrt)*maxIter)
    if out==2
    	println(@sprintf("=== gmres ===\n%4s\t%7s\n","iter","relres"))
    end
    
    flag = -1
    cnt  = 1
	iter = 1
    while iter <= maxIter
        V[:,1] = r / norm( r )
        s      = norm( r )*e1;  
        
        if out==2;; print(@sprintf("%3d\t", iter));end
        
        for i = 1:restrt
            w = A(V[:,i])
            w = M(w)
            
            for k = 1:i # basis using Gram-Schmidt
                H[k,i] = dot(w,V[:,k])
                w     -= H[k,i]*V[:,k]
            end
            H[i+1,i] = norm( w )
            V[:,i+1] = w / H[i+1,i]
            
            for k = 1:i-1 # apply Givens rotation
                temp     =  cs[k]*H[k,i] + sn[k]*H[k+1,i]
                H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i]
                H[k,i]   = temp
            end
            
            # Approximate residual norm
            cs[i],sn[i], = symOrtho( H[i,i], H[i+1,i] )
            s[i+1] = -sn[i]*s[i]
            s[i]   = cs[i]*s[i]
            H[i,i] = cs[i]*H[i,i] + sn[i]*H[i+1,i]
            H[i+1,i] = 0.0
            err  = abs(s[i+1]) / bnrm2
            
            if out==2; print(@sprintf("%1.1e ", err));end
            
            resvec[cnt] = err
            
            if err <= tol
                y  = H[1:i,1:i] \ s[1:i]
                x += V[:,1:i]*y
                if out==2; print("\n"); end
                flag = 0; break
            end
            cnt  += 1
        end
        if  err <= tol
            flag = 0
            break
        end
        y  = H[1:restrt,1:restrt]\s[1:restrt]
        x += V[:,1:restrt]*y
        if storeInterm; X[:,iter] = x; end
        
        r = b - A(x)
        r = M(r)
        
        s[restrt+1] = norm(r)
        resvec[cnt] = abs(s[restrt+1]) / bnrm2
        
        if out==2; print(@sprintf("\t %1.1e\n", err)); end
		iter += 1
    end
	iter = min(iter,maxIter)
    
    if out>=0
        if flag==-1
            println(@sprintf("gmres iterated maxIter (=%d) times without achieving the desired tolerance.",maxIter))
        elseif flag==0 && out>=1
            println(@sprintf("gmres achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[cnt]))
        end
    end
    if storeInterm
	    return X[:,1:iter],flag,resvec[cnt],iter,resvec[1:cnt]
	else
	    return x,flag,resvec[cnt],iter,resvec[1:cnt]
	end
end

