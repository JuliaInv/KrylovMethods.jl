export bicgstb

function   bicgstb(A, b::Vector; tol::Real=1e-6, maxIter::Int=100, M1=1.0, M2=1.0,x::Vector=[],out::Int=0)
# x, flag, err, iter,resvec = bicgstb(A, b, tol=1e-6, maxIter=100, M1=1.0, M2=1.0,x=[],out=0)
#
# BiConjugate Gradient Stabilized Method applied to the linear system Ax=b 
#
# Input:
#
#   A       - matrix or function computing A*x
#   b       - right hand side vector
#   tol     - error tolerance
#   maxIter - maximum number of iterations
#   M1,M2   - preconditioners, either matrices or function computing M1\x or M2\x
#   x       - starting guess
#   out     - flag for output (0 : only errors, 1 : final status, 2: relres at each iteration)
#
# Output:
#
#   x       - solution
#   flag    - exit flag (  0 : desired tolerance achieved,
#                         -1 : maxIter reached without converging
#                         -2 : rho equal to zero
#						  -3 : norm(s)/bnrm2 < tol 
# 						  -4 : omega < 1e-16
#   err     - error, i.e., norm(A*x-b)/norm(b)
#   iter    - number of iterations
#   resvec  - error at each iteration


Af(x)  =  isa(A,Function)  ? A(x) : A*x
M1f(x) =  isa(M1,Function) ? M1(x) : M1\x
M2f(x) =  isa(M2,Function) ? M2(x) : M2\x

if isempty(x)
	x = zeros(length(b))
	r = b
else
	r = b - Af(x)
end

if iseltype(b,Complex)  || iseltype(r,Complex)  || iseltype(A,Complex) 
	x = complex(x); 
end
resvec = zeros(maxIter+1)
bnrm2 = norm( b )

err = norm( r ) / bnrm2; resvec[1] = err
alpha = 1.0
omega  = 1.0
r_tld = r

iter = 1
flag = -1
if out==2
	println("=== bicgstb ===")
    println(@sprintf("%4s\t%7s","iter","relres"))
end
for iter = 1:maxIter
    rho   = dot(r_tld,r)
    if ( rho == 0.0 )
		flag = -2; break;
    end
     if ( iter > 1 )
        beta  = ( rho/rho_1 )*( alpha/omega );
        p = r + beta*( p - omega*v );
     else
        p = r;
     end

     p_hat = M1f(p)     # compute M1\p
     p_hat = M2f(p_hat) # compute M2\phat
     v = Af(p_hat)      # compute A*phat

     alpha = rho / ( dot(r_tld,v) )
     s = r - alpha*v;
     if ( norm(s)/bnrm2 < tol )
		iter -=1
		x = x + alpha*p_hat
        resid = norm( s ) / bnrm2
		flag  = -3; break 
     end
	s_hat = M1f(s)      # compute M1\s
    s_hat = M2f(s_hat)  # compute M2\shat
    t     = Af(s_hat)   # compute A*shat
    
	omega = ( dot(t,s)) / ( dot(t,t) )
    x += alpha*p_hat + omega*s_hat
    r = s - omega*t

    err = norm( r ) / bnrm2
    resvec[iter+1] = err
	if out==2
        println(@sprintf("%3d\t%1.2e",iter,resvec[iter+1]))
    end
    if ( err <= tol )
		flag = 0; break
    end

     if  norm(omega) < 1e-16
		flag = -4; break
     end

     rho_1 = rho
  end

  if flag==-1
     println(@sprintf("bicgstb iterated maxIter (=%d) times without achieving the desired tolerance.",maxIter))
  elseif flag==-2
     println(@sprintf("bicgstb: rho equal to zero at iteration %d. Returned residual has norm %1.2e.", iter,resvec[iter+1]))
  elseif flag==-3
     println(@sprintf("bicgstb : norm(s)/bnrm2 < tol."))
  elseif flag==-4
     println(@sprintf("bicgstb : omega < 1e-16"))
  elseif out>=1
     println(@sprintf("cg achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[iter+1]))
  end


  return x, flag,resvec[iter+1],iter,resvec[1:iter+1]

  end

