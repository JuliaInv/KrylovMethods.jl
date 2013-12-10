export cgls

function cgls(A,b,tol=1e-2,maxIter=100,x=[],interm=0,out=0)
# x,flag,err,iter,resvec = cg(A,b,tol=1e-2,maxIter=100,x=[],interm=0,out=0)
#
# CGLS Conjugate gradient algorithm applied implicitly to the normal equations 
# 
# 		(A'*A) x = A'*b.
#
# Input:
#
#   A       - matrix or function computing A*x = A(x,'F') and A'*x = A(x,'T')
#   b       - right hand side vector
#   tol     - error tolerance, default 1e-2
#   maxIter - maximum number of iterations, default 100
#   x       - starting guess
#   interm  - flag for returning intermediate solutions (useful in inverse 
#   out     - flag for output (0 : only errors, 1 : final status, 2: error at each iteration)
#
# Output:
#
#   x       - final solution (interm==0) or solution history (interm==1)
#   flag    - exit flag (  0 : desired tolerance achieved,
#                         -1 : maxIter reached without converging
#                         -2 : Matrix A is not positive definite )
#   eta     - residual norm: norm(A*x-b)
#   rho     - norm of current iterate: norm(x)

Af(x)  =  isa(A,Function) ? A(x,'F') : A*x
ATf(x) =  isa(A,Function) ? A(x,'T') : vec(x'*A) # use that A'*x = (x'*A)' for speedup

# Initialization.
if isempty(x)
	r = b		 	# residual r = A'*(A*x-b)
	g = ATf(r) 		# compute gradient g = A'*(A*x-b)
	n = length(g)
	x = zeros(n)
else
	n = length(x)
	r = b-Af(x)
	g = ATf(r)
end

if interm==1
	X = zeros(n,maxIter)
end

normG0 = dot(g,g)       # squared norm of gradients
normGc = normG0
eta    = zeros(maxIter) # norm of residuals
rho    = zeros(maxIter) # norm or current iterate
 
# Iterate.
iter = 1
flag = -1
if out==2
	println("=== cgls ===")
	println(@sprintf("%4s\t%8s\t%8s","iter","norm(r)","norm(x)"))
end
for iter=1:maxIter

  Ag = Af(g) # compute A*g
  
  alpha = normGc/dot(Ag,Ag)
  x  = x + alpha*g
  if interm==1
    X[:,iter] = x
  end

  r  = r - alpha*Ag
  g  = ATf(r) # compute A'*r
  
  normGt = dot(g,g)
  if normGt/normG0 < tol
	  flag = 0
      break
  end
  beta = normGt/normGc
  g = g + beta*g

  # store intermediates and report resuls
  normGc = normGt
  rho[iter] = norm(r)
  eta[iter] = norm(x)
  if out==2
  	println(@sprintf("%3d\t%1.2e\t%1.2e", iter,rho[iter],eta[iter]))
  end
end

if flag==-1
   println(@sprintf("cgls iterated maxIter (=%d) times but reached only norm %1.2e instead of tol=%1.2e.",maxIter,normGc/normG0,tol))
elseif out>=1
   println(@sprintf("cgls achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,normGc/normG0))
end

if interm==1
	return X[:,1:iter],flag,rho,eta
else
	return x,flag,rho,eta
end

end
