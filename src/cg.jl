export cg

function cg(A,b::Vector; tol::Real=1e-2,maxIter::Int=100,M=x->x ,x::Vector=[],out::Int=0)
# x,flag,err,iter,resvec = cg(A,b,tol=1e-2,maxIter=100,M=1,x=[],out=0)
#
# (Preconditioned) Conjugate Gradient applied to the linear system A*x = b, where A is assumed 
# to be symmetric positive semi definite.
#
# Input:
#
#	A       - matrix or function computing A*x
#	b       - right hand side vector
#	tol     - error tolerance
#	maxIter - maximum number of iterations
#	M       - preconditioner, either matrix or function computing M\x
#	x       - starting guess
#	out     - flag for output (-1: no output, 0: only errors, 1: final status, 2: residual norm at each iteration)
#
# Output:
#
#	x       - approximate solution
#	flag    - exit flag ( 0 : desired tolerance achieved,
#						 -1 : maxIter reached without converging
#						 -2 : Matrix A is not positive definite
#						 -3 : cg stalled, i.e. two consecutive residuals have same norm)
#	err     - norm of relative residual, i.e., norm(A*x-b)/norm(b)
#	iter    - number of iterations
#	resvec  - norm of relative residual at each iteration
	
	Af =  isa(A,Function) ? A : x->A*x
	Mf =  isa(M,Function) ? M : x->M\x
	
	if isempty(x)
		x = zeros(size(b,1))
		r = b
	else
		r = b - Af(x)
	end
	
	z = Mf(r)
	p = z
	nr0  = norm(b)
	
	if out==2
		println("=== cg ===")
		println(@sprintf("%4s\t%7s","iter","relres"))
	end
	
	resvec = zeros(maxIter)
	iter   = 1 # makes iter available outside the loop
	flag   = -1
	for iter=1:maxIter
		Ap = Af(p)
		gamma = dot(r,z)
		alpha = gamma/dot(p,Ap)
		if alpha==Inf || alpha<0
			flag = -2; break
		end
		x += alpha*p
		r -= alpha*Ap
		resvec[iter]  = norm(r)/nr0
		if out==2
			println(@sprintf("%3d\t%1.2e",iter,resvec[iter]))
		end
		if resvec[iter] <= tol
			flag = 0; break
		end
		
		z    = Mf(r)
		beta = dot(z,r)/gamma
		p    = z + beta*p
	end
	
	if out>=0
		if flag==-1
			println(@sprintf("cg iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",
																								maxIter,resvec[iter],tol))
		elseif flag==-2
			println("Matrix A in cg has to be positive definite.")
		elseif flag==0 && out>=1
			println(@sprintf("cg achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[iter]))
		end
	end
	return x,flag,resvec[iter],iter,resvec[1:iter]
end
