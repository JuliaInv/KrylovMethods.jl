export cg

function cg(A,b::Vector; tol::Real=1e-2,maxIter::Int=100,M=1,x::Vector=[],out::Int=0)
# x,flag,err,iter,resvec = cg(A,b,tol=1e-2,maxIter=100,M=1,x=[],out=0)
#
# (Preconditioned) Conjugate Gradient applied to the linear system A*x = b
#
# Input:
#
#   A       - matrix or function computing A*x
#   b       - right hand side vector
#   tol     - error tolerance
#   maxIter - maximum number of iterations
#   M       - preconditioner, either matrix or function computing M\x
#   x       - starting guess
#   out     - flag for output (0 : only errors, 1 : final status, 2: error at each iteration)
#
# Output:
#
#   x       - solution
#   flag    - exit flag (  0 : desired tolerance achieved,
#                         -1 : maxIter reached without converging
#                         -2 : Matrix A is not positive definite )
#   err     - error, i.e., norm(A*x-b)/norm(b)
#   iter    - number of iterations
#   resvec  - error at each iteration    resvec = zeros(maxIter)

resvec = zeros(maxIter)
Af(x) =  isa(A,Function) ? A(x) : A*x
Mf(x) =  isa(M,Function) ? M(x) : M\x
if isempty(x)
    x = zeros(size(b,1))
    r = b
else
    r = b - Af(x)
end
z = Mf(r)
p = z
nr0  = norm(b)

flag = 0
iter = -1

if out==2
    println("=== cg ===")
	println(@sprintf("%4s\t%7s","iter","relres"))
end

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
    if resvec[iter] < tol
        flag = 0; break
    end
    z = Mf(r)

    beta = dot(z,r)/gamma
    p = z + beta*p
end

if flag==-1
   println(@sprintf("cg iterated maxIter (=%d) times without achieving the desired tolerance.",maxIter))
elseif flag==-2
   println("Matrix A in cg has to be positive definite.")
elseif out>=1
   println(@sprintf("cg achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[iter]))
end
return x,flag,resvec[iter],iter,resvec[1:iter]
end

