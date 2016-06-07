export blockCG

function blockCG{T1,T2}(A::SparseMatrixCSC{T1,Int},B::Array{T2,2}; kwargs...) 
	X = zeros(promote_type(T1,T2),size(A,2),size(B,2)) # pre-allocate
	return blockCG(V -> At_mul_B!(1.0,A,V,0.0,X),B;kwargs...) # multiply with transpose of A for efficiency
end

blockCG(A,B::Array;kwargs...) = blockCG(X -> A*X,B;kwargs...)

"""
blockCG(A,B,X)

Preconditioned Conjugate Gradient Method for solving

A * X = B

Input:

	A           - matrix or function computing A*x
	B           - array of right hand sides
	tol         - error tolerance
	maxIter     - maximum number of iterations
	M           - preconditioner, a function that computes M\\x
	X           - array of starting guesses (will be overwritten)
	out         - flag for output (-1: no output, 0: only errors, 1: final status, 2: residual norm at each iteration)
	ortho       - flag for re-orthogonalization (default: false)
	pinvTol     - tolerance for pseudoinverse (default: eps(T)*size(B,1))
	storeInterm - flag for storing iterates (default: false)

Output:

  X       - approximate solutions (2D array if !storeInterm, 3D array else)
  flag    - exit flag ( 0 : desired tolerance achieved,
  					 -1 : maxIter reached without converging
  					 -9 : right hand side was zero)
  err     - norm of relative residual, i.e., norm(A*X-B)/norm(B)
  iter    - number of iterations
  resvec  - norm of relative residual at each iteration


Reference: 

	O'Leary, D. P. (1980). The block conjugate gradient algorithm and related methods. 
	Linear Algebra and Its Applications, 29, 293–322. http://doi.org/10.1016/0024-3795(80)90247-5

"""
function blockCG{T<:AbstractFloat}(A::Function,B::Array{T};X=zeros(T,size(B)),M::Function=identity,maxIter=20,tol=1e-2,ortho::Bool=false,pinvTol =eps(T)*size(B,1),out::Int=0,storeInterm::Bool=false)

if norm(B)==0; return zeros(eltype(B),size(B)),-9,0.0,0,[0.0]; end
	
R = copy(B)
Z = M(R)
if ortho
    P, = mgs(Z)
else
    P = Z
end
 
n, nrhs = size(X)
nB      = computeNorm(B)
resmat  = zeros(maxIter,nrhs) 

if storeInterm
    Xout = zeros(n,nrhs,maxIter)	# allocate space for intermediates
end

if out==2
	println("=== blockcg ===")
	println(@sprintf("%4s\t%7s","iter","max(relres)"))
end

# pre-allocate
PTQ = zeros(T,nrhs,nrhs)
PTR = zeros(T,nrhs,nrhs)
QTZ = zeros(T,nrhs,nrhs)

flag = -1;
iter = 0
for iter=1:maxIter
	Q = A(P)
    # PTQ   = P'*Q
	BLAS.gemm!('T','N',1.0,P,Q,0.0,PTQ)

    # Alpha = (PTQ)\(P'*R);
    BLAS.gemm!('T','N',1.0,P,R,0.0,PTR)
	pinvPTQ = getPinv!(PTQ,pinvTol)
    Alpha = pinvPTQ*PTR
    
    # X     += P*Alpha
    BLAS.gemm!( 'N','N',1.0, P, Alpha, 1.0, X)
	if storeInterm; Xout[:,:,iter] = X; end
    # R     -= Q*Alpha
    BLAS.gemm!('N','N',-1.0,Q,Alpha,1.0,R)

    resmat[iter,:] = computeNorm(R)./nB
    if out==2
		println(@sprintf("%3d\t%1.2e",iter,maximum(resmat[iter,:])))
	end
    if maximum(resmat[iter,:]) < tol
	    flag = 0;
        break;
    end
    
    Z     = M(R)
    #Beta  = -(PTQ)\(Q'*Z);
	BLAS.gemm!('T','N',1.0,Q,Z,0.0,QTZ)
    Beta  = -pinvPTQ*QTZ
    
    if ortho
        P,     =  mgs!(Z + P*Beta)
    else
		BLAS.gemm!('N','N',1.0,P,Beta,1.0,Z);
		P[:] = Z;
	    # P     = Z  + P*Beta;
	end
end
if out>=0
	if flag==-1
		println(@sprintf("blockCG iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",																							maxIter,maximum(resmat[iter,:]),tol))
	elseif flag==0 && out>=1
		println(@sprintf("blockCG achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,maximum(resmat[iter,:])))
	end
end
if storeInterm
    return Xout[:,:,1:iter],flag,resmat[iter,:],iter,resmat[1:iter,:]
else
    return X,flag,resmat[iter,:],iter,resmat[1:iter,:]
end
end

function computeNorm(R)
	n,nrhs = size(R)
	res    = zeros(nrhs)
	for k=1:nrhs
		for i=1:n
			res[k]+=R[i,k]*R[i,k]
		end
	end
	return sqrt(res)
end

function getPinv!(A,pinvTol)
	SVD         = svdfact!(A)
	Sinv        = zeros(length(SVD.S))
    index       = SVD.S .> pinvTol*maximum(SVD.S)
    Sinv[index] = 1.0./ SVD.S[index]
    Sinv[find(!isfinite(Sinv))] = 0.0
    return SVD.Vt'scale(Sinv, SVD.U')
end
 
