export bicgstb

function bicgstb(A::SparseMatrixCSC{T1,Int},b::Array{T2,1}; kwargs...) where {T1,T2}
	Ax = zeros(promote_type(T1,T2),size(A,2))                  # pre-allocate
	# return bicgstb(x -> A_mul_B!(1.0,A,x,0.0,Ax),b;kwargs...) # multiply with transpose of A for efficiency
	return bicgstb(x -> mul!(Ax,A,x,1.0,0.0),b;kwargs...) # multiply with transpose of A for efficiency
end

bicgstb(A,b; kwargs...) =  bicgstb(x -> A*x,b; kwargs...)

"""
x,flag,err,iter,resvec = bicgstb(A,b,tol=1e-6,maxIter=100,M1=1.0,M2=1.0,x=[],out=0)

BiConjugate Gradient Stabilized Method applied to the linear system Ax=b. 

Input:

  A       - computing A*x
  b       - right hand side vector
  tol     - error tolerance
  maxIter - maximum number of iterations
  M1,M2   - preconditioners, either matrices or functions computing M1\\x or M2\\x
  x       - starting guess
  out     - flag for output (-1: no output, 0 : only errors, 1 : final status, 2: relres at each iteration)

Output:

  x       - solution
  flag    - exit flag (  	0 : desired tolerance achieved,
  						-1 : maxIter reached without converging
  						-2 : rho equal to zero
  						-3 : norm(s)/bnrm2 < tol 
  						-4 : omega < 1e-16
  						-9 :  right hand side was zero)
  err     - error, i.e., norm(A*x-b)/norm(b)
  iter    - number of iterations
  resvec  - error at each iteration
"""
function bicgstb(A::Function, b::Vector; tol::Real=1e-6, maxIter::Int=100, M1=x->copy(x), M2=x->copy(x),x::Vector=[],out::Int=0,storeInterm::Bool=false, tolRho::Real=1e-40)

	n   = length(b)
	if norm(b)==0; return zeros(eltype(b),n),-9,0,0,zeros(1); end
	M1f =  isa(M1,Function) ? M1 : x -> M1\x
	M2f =  isa(M2,Function) ? M2 : x -> M2\x
	
	# allocate v,t,p_hat,s_hat
	v     = zeros(eltype(b),n)
	t     = zeros(eltype(b),n)
	p_hat = zeros(eltype(b),n)
	s     = zeros(eltype(b),n)
	s_hat = zeros(eltype(b),n)

	if isempty(x)
		x = zeros(eltype(b),n)
		r = copy(b)
	else
		r = b - A(x)
	end
    if storeInterm
        X = zeros(eltype(b),n,maxIter)	# allocate space for intermediates
    end
	
	resvec = zeros(maxIter+1)
	bnrm2 = norm( b )
	resid = norm( r ) / bnrm2; resvec[1] = resid
	alpha = 1.0
	omega = 1.0
	r_tld = copy(r)
	
	flag = -1
	rho1 = 0.0
	p   = copy(r)
	if out==2
		println("=== bicgstb ===")
		println(@sprintf("%4s\t%7s","iter","relres"))
	end
	iter = 1
	while iter <= maxIter
		rho   = dot(r_tld,r)
		if ( abs(rho) < tolRho ); flag = -2; break; end
		
		if ( iter > 1 )
			beta  = ( rho/rho1 )*( alpha/omega )
			# p = r + beta*( p - omega*v );
			BLAS.scal!(n,beta,p,1)
			BLAS.axpy!(n,-(beta*omega),v,1,p,1)
			BLAS.axpy!(n,one(eltype(p)),r,1,p,1)
		end
		
		p_hat = M1f(p)      # compute M1\p
		p_hat = M2f(p_hat)  # compute M2\phat
		t     = A(p_hat)    # compute A*phat
		v[:] = t;
		
		alpha = rho / ( dot(r_tld,v) )
		
		BLAS.axpy!(n,alpha,p_hat,1,x,1) # x = x + alpha*p_hat
		BLAS.axpy!(n,-alpha,v,1,r,1)	# r = r - alpha*v
		resid = norm(r)/bnrm2;
		if ( resid < tol )
			iter -=1

			flag  = -3; break 
		end	
		
		# p_hat must not be used beyond this point because M2 or M1 might be using the same memory for s_hat.
		s_hat = M1f(r)      # compute M1\s
		s_hat = M2f(s_hat)  # compute M2\shat
		t     = A(s_hat)    # compute A*shat
		omega = ( dot(t,r)) / ( dot(t,t) )
		
		
		BLAS.axpy!(n,-omega,t    ,1,r,1) # r = r - omega * t
		BLAS.axpy!(n,omega ,s_hat,1,x,1) # x = x + omega * s_hat
		
		resid = norm( r ) / bnrm2
		resvec[iter+1] = resid
		if storeInterm; X[:,iter] = x; end
		if out==2
			println(@sprintf("%3d\t%1.2e",iter,resvec[iter+1]))
		end
		if ( resid <= tol )
			flag = 0; break
		end
		
		if norm(omega) < 1e-16; flag = -4; break; end
		rho1 = rho
		iter += 1
	end
	iter = min(iter,maxIter)
	
	if out>=0
		if flag==-1
			println(@sprintf("bicgstb iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",
																								maxIter,resvec[iter],tol))
		elseif flag==-2
			println(@sprintf("bicgstb: rho equal to zero at iteration %d. Returned residual has norm %1.2e.", iter,resvec[iter]));
		elseif flag==-4
			println(@sprintf("bicgstb : omega < 1e-16"))
		elseif out>=1
			if flag == -3
				println(@sprintf("bcgstb achieved desired tolerance at iteration %d.5. Residual norm is %1.2e.",iter,resid))
			else
				println(@sprintf("bcgstb achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[iter+1]))
			end
		end
	end
	if storeInterm
		return X[:,1:iter], flag,resvec[iter+1],iter,resvec[1:iter+1]
	else
		return x, flag,resvec[iter+1],iter,resvec[1:iter+1]
	end
end
