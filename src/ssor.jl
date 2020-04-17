export ssor, sorPrec!, ssorPrecTrans!

"""
x,flag,err,iter,resvec = ssor(A::SparseMatrixCSC, b::Vector; omega::Real=2/3, tol::Real=1e-2, maxIter::Int=1,out::Int=0)

Symmetric successive over-relaxation applied to the linear system A*x = b.

!! Note that upper/lower triangular matrix are never build to save memory !!

Input:

	A       - sparse matrix CSC
	b       - right hand side vector
	omega   - relaxation parameter (omega=1.0: Gauss Seidel)
	tol     - error tolerance (default=1e-2)
	maxIter - maximum number of iterations (default=1)
	out     - flag for output (-1 : no output, 0 : only errors, 1 : final status, 2: error at each iteration)

Output:

	x       - solution
	flag    - exit flag (  0 : desired tolerance achieved,
	                      -1 : maxIter reached without converging)
	err     - error, i.e., norm(A*x-b)/norm(b)
	iter    - number of iterations
	resvec  - error at each iteration
"""
function ssor(A::SparseMatrixCSC,b::Vector;x::Vector=[],tol::Real=1e-2,maxIter::Int=1,omega::Real=1.0,out::Int=0,storeInterm::Bool=false)
n      = length(b)
OmInvD = Array(omega./diag(A))

if storeInterm
    X = zeros(n,maxIter)	# allocate space for intermediates
end

if isempty(x)
	x = zeros(eltype(A),n)
	r = copy(b)
	rnorm0 = norm(b)
else
	r = b - A*x
	rnorm0 = norm(r)
end
if out==2
	println(@sprintf("=== sor ===\n%4s\t%7s\n","iter","relres"))
end

resvec = zeros(maxIter)
flag   = -1
iter   = 1
while iter <= maxIter
	ssorPrec!(A,x,r,OmInvD)
	if storeInterm; X[:,iter] = x; end
	resvec[iter] = norm(r)/rnorm0
	if out==2; println(@sprintf("%3d\t%1.2e",iter,resvec[iter]));end
	if resvec[iter] < tol
		flag=0; break
	end
	iter += 1
end
iter = min(iter,maxIter)

if (flag==-1) && (out>=0)
	println(@sprintf("ssor iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",maxIter,resvec[iter],tol))
elseif  (out>=1)
	println(@sprintf("ssor achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[iter]))
end
if storeInterm
	return X[:,1:iter],flag,resvec[iter],iter,resvec[1:iter]
else
	return x,flag,resvec[iter],iter,resvec[1:iter]
end	
end




"""
 ssorPrec!(A::SparseMatrixCSC,x::Vector,r::Vector,OmInvD::Vector=1./diag(A))

 Applies one SSOR step to A*x = r and updates the residual.        

 Input:

	A       - sparse matrix CSC
	x       - current iterate
	r       - residual 
	OmInvD  - weighted diagional, that is, omega./diag(A)

 Output: This method changes x and r on the fly.

 see also sor, which uses ssorPrec! in each iteration.
 WARNING: DO NOT USE AS PRECONDITIONER, SINCE THIS METHOD OVEWRITES THE VECTOR r
"""
function ssorPrec!(A::SparseMatrixCSC,x::Vector,r::Vector,OmInvD::Vector=Array(1 ./diag(A)))

dx = 0.0
n  = length(x)

for j=n:-1:1
	dx = OmInvD[j]
	dx *= r[j]
	for gidx = A.colptr[j] : A.colptr[j+1]-1
		r[A.rowval[gidx]] -= A.nzval[gidx]*dx
	end
	x[j] += dx
end
for j=1:n
	dx  = OmInvD[j]
	dx *= r[j]
	for gidx = A.colptr[j] : A.colptr[j+1]-1
		r[A.rowval[gidx]] -= A.nzval[gidx]*dx
	end
	x[j] += dx
end
return x
end

"""
x = ssorPrecTrans!(A::SparseMatrixCSC,X::Array,B::Array,OmInvD::Vector=1 ./diag(A))

Applies SSOR steps to the linear systems A*X = B, for symmetric A. For efficiency, it works on the
transpose of the matrix A.       

Input:

  A       - sparse matrix CSC, assumed to be symmetric here
  X       - current iterates
  R       - residuals 
  OmInvD  - weighted diagional, that is, omega./diag(A)

Output: This method changes X and on the fly. The Array B is unchanged.

This version is well-suited as a preconditioner, for example, for CG.

Example for solving A*X=B, when A is Sparse Symmtric Positive definite:
    omega = 1.2;
    d = omega./diag(A);
    x = zeros(length(d)) # pre allocation for the preconditioner result.
    PC(r) = (x[:]=0.0; return ssorPrecTrans!(A,x,r,d));
    y = KrylovMethods.cg(A,b,tol=1e-12,maxIter=200,M=PC,out=1)[1]
"""
function ssorPrecTrans!(A::SparseMatrixCSC,X::Array,B::Array,OmInvD::Vector=Array(1 ./diag(A)))

r    = 0.0
n    = size(X,1)
nrhs = size(X,2)

for k=1:nrhs
	for j=n:-1:1
		r = B[j,k]
		for gidx = A.colptr[j] : A.colptr[j+1]-1
			r -= A.nzval[gidx]*X[A.rowval[gidx],k]
		end
		r *= OmInvD[j]
		X[j,k] += r
	end
	for j=1:n
		r = B[j,k]
		for gidx = A.colptr[j] : A.colptr[j+1]-1
			r -= A.nzval[gidx]*X[A.rowval[gidx],k]
		end
		r *= OmInvD[j]
		X[j,k] += r
	end
end
return X
end

