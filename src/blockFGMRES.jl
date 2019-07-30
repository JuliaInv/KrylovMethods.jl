
import Base.isempty
using LinearAlgebra
export blockFGMRES

function blockFGMRES(A::SparseMatrixCSC{T1,Int},b::Array{T2,2},restrt::Int; kwargs...) where {T1,T2}
	Ax = zeros(promote_type(T1,T2),size(b))
	return blockFGMRES(X -> mul!(Ax,A,X,1.0,0.0),b,restrt;kwargs...)
end

blockFGMRES(A,b::Array{Any,2},restrt;kwargs...) = blockFGMRES(x -> A*x ,b,restrt;kwargs...) 


"""
x,flag,err,iter,resvec = blockFGMRES(A,b,restrt,tol=1e-2,maxIter=100,M=1,x=[],out=0)

Block (flexible) Generalized Minimal residual ( (F)GMRES(m) ) method with restarts applied to A*x = b.

This is the "right preconditioning" version of blockGMRES: A*M^{-1}Mx = b. 

Input:

A       - function computing A*x
b       - right hand side matrix (set of vectors)
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
	
	preconditioner M(r) must return a copy of a matrix. Cannot reuse memory of r.
	"""
function blockFGMRES(A::Function,B::Array,restrt::Int; tol::Real=1e-2,maxIter::Int=100,M::Function=t->copy(t),X::Array=[],out::Int=0,flexible::Bool=false,mem::FGMRESmem =  getEmptyFGMRESmem())
  # initialization
  n  = size(B,1)
  m  = size(B,2)
  TYPE = eltype(B)
  mem = checkMemorySize(mem,n,restrt,TYPE,flexible,m);

  if norm(B)==0.0 
	return zeros(TYPE,n,m),-9,0.0,0,[0.0]; 
  end
	
  if Base.isempty(X)
    X = zeros(eltype(B),n,m)
	R = copy(B);
  elseif norm(X) < eps(real(TYPE))
	R = copy(B);
  else
	R = copy(B);
	R.-=A(X);
  end
  
  if eltype(B) <: Complex 
       X = complex(X)
   end
  if issparse(X)
	error("X is sparse");
  end
  
  rnorm0 = norm(B);

  err = norm(R)/rnorm0
  if err < tol; return X, err; end

  constOne = one(TYPE);
  constZero = zero(TYPE);

  restrt = min(restrt,n-1)
  Zbig = mem.Z;
  Vbig = mem.V;
  H = zeros(TYPE,(restrt+1)*m,restrt*m);
  xi = zeros(TYPE,(restrt+1)*m,m);
  T = zeros(TYPE,restrt*m,m);
 
  W = zeros(TYPE,0);
  Z = zeros(TYPE,0);
  
  resvec = zeros(restrt*maxIter)
  if out==2
    println(@sprintf("=== blockFGMRES ===\n%4s\t%7s\n","iter","relres"))
  end

  
  flag = -1
	counter = 0
	iter = 0
	while iter < maxIter
		iter+=1;
		Q = qr!(R); 
		Betta = Q.R; 
		R[:] = Matrix(Q.Q); # this code is equivalent to (W,Betta) = qr(r);
		xi[1:m,:] = Betta;
		#BLAS.scal!(n,(1/betta)*constOne,r,1); # w = w./betta
		H[:] .= 0.0;
		T[:] .= 0.0;
		
		if out==2;; print(@sprintf("%3d\t", iter));end
		
		for j = 1:restrt
			colSet_j = (((j-1)*m)+1):j*m
			if j==1
				Vbig[:,colSet_j] = R;  # no memory problem with this line....
				Z = M(R); 
			else
				Vbig[:,colSet_j] = W;  # no memory problem with this line....
				Z = M(W);
			end
			if flexible
				Zbig[:,colSet_j] = Z;
			end
			W = A(Z);
			counter += 1;
			# Gram Schmidt (much faster than MGS even though zeros are multiplied, does relatively well):
			BLAS.gemm!('C','N', constOne, Vbig, W,constZero,T); # t = V'*w;
			T[(j*m+1):end,:] .= 0.0;
			H[1:(restrt*m),colSet_j] = T;
			BLAS.gemm!('N','N', -constOne, Vbig, T,constOne,W); # w = w - V*t  
			Q = qr!(W); Betta = Q.R; W[:] = Matrix(Q.Q); # this code is equivalent to (W,Betta) = qr(W);
			H[((j*m)+1):(j+1)*m,colSet_j] = Betta;
			y = H[1:(j+1)*m,1:j*m]\xi[1:(j+1)*m,:];
			err = norm(H[1:(j+1)*m,1:j*m]*y - xi[1:(j+1)*m,:])/rnorm0
		
			if out==2 print(@sprintf("%1.1e ", err)); end
			resvec[counter] = err;
			if err <= tol
				if flexible
					Zbig[:,(j*m+1):end] .= 0.0;
				else
					Vbig[:,(j*m+1):end] .= 0.0;
				end
				if out==2; print("\n"); end
				flag = 0; break
			end
		end # end for j to restrt
		
		y = pinv(H)*xi;
		
		if flexible
			BLAS.gemm!('N','N', constOne, Zbig, y,constZero,W); # w = Z*y  #This is the correction that corresponds to the residual.
		else
			# W = Vbig*y;
			BLAS.gemm!('N','N', constOne, Vbig, y, constZero, W);
			Z = M(W);
			W[:] = Z;
		end
		X .+= W;
		
		if out==2; print("\n"); end
		if err <= tol
			flag = 0;
			break
		end
		if iter < maxIter
			R[:] = B;
			W = A(X);
			R .-= W;
		end
	end #for iter to maxiter
	if out>=0
		if flag==-1
			println(@sprintf("blockFGMRES iterated maxIter (=%d) outer iterations without achieving the desired tolerance. Acheived: %1.2e",maxIter,err))
		elseif flag==0 && out>=1
			println(@sprintf("blockFGMRES achieved desired tolerance at inner iteration %d. Residual norm is %1.2e.",counter,resvec[counter]))
		end
	end
	return X,flag,resvec[counter],iter,resvec[1:counter]
end #blockFGMRES
