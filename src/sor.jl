export sor, fwdTriSolveOmega, bwdTriSolveOmega

function fwdTriSolveOmega(A::SparseMatrixCSC,b::Vector,omega::Real=1)
# x = fwdTriSolveOmega(A::SparseMatrixCSC,b::Vector,omega::Real=1)
#
# Solves ((1./omega)*diag(A)+tril(A,1)) * x = b by forward substitution.
#
#  !! Note that (omega*diag(A)+triu(A,1)) is never build even if A is not upper triangular !!
# 
# Input:
#
#   A       - square sparse matrix (not necessarily upper triangular)
#   b       - right hand side vector
#   omega   - weighting of diagonal (1: yields standard triangular solve)
#
# Output:
#
#   x       - solution


omegaInv = 1.0/omega
n = length(b)
ncol, nrow = size(A)
if ncol!=nrow; error("A must be square"); end;
if n!=ncol; error("A must be nXn"); end;

x = zeros(n)
if iseltype(b,Complex); x = complex(x); end

aa = A.nzval
ja = A.rowval
ia = A.colptr

for j=1:n
   i1 = ia[j]
   i2 = ia[j+1]-1
   for i = i1:i2
      if ja[i] == j
         x[j] = (b[j]-x[j])/(omegaInv*aa[i])
      elseif ja[i] > j
         x[ja[i]] += aa[i]*x[j]
      end
   end
end
return x
end


function bwdTriSolveOmega(A::SparseMatrixCSC,b::Vector,omega::Real=1)
# x = bwdTriSolveOmega(A::SparseMatrixCSC,b::Vector,omega::Real=1)
#
# Solves ((1./omega)*diag(A)+triu(A,-1)) * x = b by backward substitution.
#
#  !! Note that (omega*diag(A)+tril(A,-1)) is never build even if A is not upper triangular !!
# 
# Input:
#
#   A       - square sparse matrix (not necessarily lower triangular)
#   b       - right hand side vector
#   omega   - weighting of diagonal (1: yields standard triangular solve)
#
# Output:
#
#   x       - solution

n = length(b)
ncol, nrow = size(A)
if ncol!=nrow; error("A must be square"); end;
if n!=ncol; error("A must be nXn"); end;

x = zeros(n)
if iseltype(b,Complex); x = complex(x); end

omegaInv = 1.0/omega
aa = A.nzval
ja = A.rowval
ia = A.colptr

for j=n:-1:1
   i1 = ia[j]
   i2 = ia[j+1]-1
   for i = i2:-1:i1
       if ja[i]==j
          x[j] = (b[j]-x[j])/(omegaInv*aa[i])
       elseif ja[i]<j
          x[ja[i]] += aa[i]*x[j]
       end
   end
end
return x
end



function sor(A::SparseMatrixCSC, b::Vector; omega::Real=1, tol::Real=1e-2, maxIter::Int=1,x::Vector=[],out::Int=0)
# x,flag,err,iter,resvec = sor(A::SparseMatrixCSC, b::Vector; omega::Real=2/3, tol::Real=1e-2, maxIter::Int=1,out::Int=0)
#
# Successive over-relaxation applied to the linear system A*x = b 
#
# !! Note that upper/lower triangular matrix are never build (--> memory efficient :) )!!
#
# Input:
#
#   A       - sparse matrix CSC
#   b       - right hand side vector
#   omega   - relaxation parameter (omega=1: Gauss Seidel)
#   tol     - error tolerance
#   maxIter - maximum number of iterations
#   x       - starting guess
#   out     - flag for output (0 : only errors, 1 : final status, 2: error at each iteration)
#
# Output:
#
#   x       - solution
#   flag    - exit flag (  0 : desired tolerance achieved,
#                         -1 : maxIter reached without converging)
#   err     - error, i.e., norm(A*x-b)/norm(b)
#   iter    - number of iterations
#   resvec  - error at each iteration

resvec = zeros(maxIter)
n    = size(A,1)
x    = zeros(n)
dx   = zeros(n)
r    = zeros(n)
for i=1:n; r[i] = b[i]; end;
r0   = norm(b)
d    = (1/omega)*diag(A)
iter = 1

if out==2
    println("=== sor ===")
	println(@sprintf("%4s\t%7s","iter","relres"))
end
flag = -1
for iter=1:maxIter
	dx = bwdTriSolveOmega(A,r,omega)
	dx = dx.*d
	dxOld = dx
	dx = fwdTriSolveOmega(A,dxOld,omega)
	dx = (2-omega)*dx
	
	x += dx
	r = b - A*x
	
	resvec[iter] = norm(r)/r0
    if out==2
        println(@sprintf("%3d\t%1.2e",iter,resvec[iter]))
    end
	if resvec[iter] <= tol
		flag=0; break
	end
end
if flag==-1
   println(@sprintf("sor iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",resvec[iter],maxIter,tol))
elseif  out==1
   println(@sprintf("sor achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[iter]))
end

return x,flag,resvec[iter],iter,resvec[1:iter]
end