export cg


function cg(A,b::Vector; tol::Real=1e-2,maxIter::Int=100,M=x->x ,x::Vector=[],out::Int=0)
# x,flag,err,iter,resvec = cg(A,b,tol=1e-2,maxIter=100,M=1,x=[],out=0)
#
# (Preconditioned) Conjugate Gradient applied to the linear system A*x = b, where A is assumed 
# to be symmetric positive semi definite.
#
# Input:
#
#   A       - matrix or function computing A*x
#   b       - right hand side vector
#   tol     - error tolerance
#   maxIter - maximum number of iterations
#   M       - preconditioner, either matrix or function computing M\x
#   x       - starting guess
#   out     - flag for output (0: only errors, 1: final status, 2: residual norm at each iteration)
#
# Output:
#
#   x       - approximate solution
#   flag    - exit flag (  0 : desired tolerance achieved,
#                         -1 : maxIter reached without converging
#                         -2 : Matrix A is not positive definite
#  						  -3 : cg stalled, i.e. two consecutive residuals have same norm)
#   err     - norm of relative residual, i.e., norm(A*x-b)/norm(b)
#   iter    - number of iterations
#   resvec  - norm of relative residual at each iteration

resvec = zeros(maxIter)
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

flag = -1

if out==2
    println("=== cg ===")
	println(@sprintf("%4s\t%7s","iter","relres"))
end

iter = 1 # makes iter available outside the loop
for iter=1:maxIter
    Ap = Af(p)
    gamma = dot(r,z)
    alpha = gamma/dot(p,Ap)
    if alpha == Inf || alpha <0
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
    elseif (iter>1) && (resvec[iter] == resvec[iter-1])
		flag = -3; break
	end
	
    z = Mf(r)

    beta = dot(z,r)/gamma
    p = z + beta*p
end

if flag==-1
   println(@sprintf("cg iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",resvec[iter],maxIter,tol))
elseif flag==-2
   println("Matrix A in cg has to be positive definite.")
elseif flag==-3
   println("cg stalled: Two consecutive residuals have same norm.")
elseif out>=1
   println(@sprintf("cg achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[iter]))
end
return x,flag,resvec[iter],iter,resvec[1:iter]
end

