
export BlockBiCGSTB

function BlockBiCGSTAB{T1,T2}(A::SparseMatrixCSC{T1,Int},b::Array{T2,2}; kwargs...) 
	TYPE = promote_type(T1,T2);
	Ax = zeros(TYPE,size(b));                
	return BlockBiCGSTB(x -> A_mul_B!(one(TYPE),A,x,zero(TYPE),Ax),b;kwargs...); 
end

BlockBiCGSTB(A,b; kwargs...) =  BlockBiCGSTB(x -> A*x,b; kwargs...)

function BlockBiCGSTB{T}(A::Function, b::Array{T}; tol::Real=1e-6, maxIter::Int=100, M1=x->copy(x), M2=x->copy(x),x::Array=[],out::Int=0)

	n   = size(b,1);
	m   = size(b,2);
	
	N = m*n;
	
	if vecnorm(b)==0; 
		return zeros(eltype(b),n,m),-9; 
	end
	M1f =  isa(M1,Function) ? M1 : x -> M1\x
	M2f =  isa(M2,Function) ? M2 : x -> M2\x
	
	constOne = one(T);
	constZero = zero(T);
	
	if length(x)==0
		x = zeros(eltype(b),n,m)
		r = copy(b)
		r_tld = b; # We are not changing r_tld in this function, so we assign it to point to b that is also held constant.
	elseif vecnorm(x)==0.0
		r = copy(b)
		r_tld = b; # We are not changing r_tld in this function, so we assign it to point to b that is also held constant.
	else
		r = b - A(x);
		r_tld = copy(r); # We are not changing r_tld in this function, but we do change r;
	end
	v 	= zeros(T,n,m);
    p   = copy(r);   
	
	resvec = zeros(maxIter+1)
	bnrm2 = vecnorm( b )
	
	rnorm   = vecnorm( r ) / bnrm2; 
	resvec[1] = rnorm
	
	alpha = 1.0
	omega = 1.0
	
	iter = 1
	flag = -1
	rho1 = 0.0
	
	if out==2
		println("=== block bicgstb ===")
		println(@sprintf("%4s\t%7s","iter","relres"))
	end
	RtV = [];
	t = [];
	for iter = 1:maxIter
		rho = BLAS.gemm('C','N', constOne, r_tld, r); 				# equivalent to rho   = r_tld'*r;
		
		if ( vecnorm(rho) < 1e-13 ); flag = -2; break; end
		
		if ( iter > 1 )
			RtT = BLAS.gemm('C','N', constOne, r_tld, t); # RtT  = r_tld'*t;
			beta = - RtV \ RtT;
			# the next lines do: p = r + (p-v*omega)*beta; We run over v here...
			BLAS.scal!(N,-omega,v,1); 									# v = -v*omega
			BLAS.axpy!(N,constOne,p,1,v,1); 							# v += p;
			BLAS.gemm!('N','N', constOne, v, beta,constZero,p); 		# p = v*beta;
			BLAS.axpy!(N,constOne,r,1,p,1); 							# p = p + r
		end

		p_hat = M1f(p);      # compute M1\p
		p_hat = M2f(p_hat) ; # compute M2\phat

		t     = A(p_hat);    # compute A*phat

		v[:]  = t;
		
		RtV = BLAS.gemm('C','N', constOne, r_tld, v); 					# RtV = r_tld'*v;
		alpha = RtV\rho; 
		BLAS.gemm!('N','N', constOne, p_hat, alpha,constOne,x); 		# x = x + p_hat*alpha # After this line p_hat is done.
		BLAS.gemm!('N','N', -constOne, v, alpha,constOne,r);  			# r = r - v*alpha;

		resid = vecnorm( r ) / bnrm2
		if ( resid < tol )
			iter -=1	
			flag  = -3; break 
		end
		
		s_hat = M1f(r)      # compute M1\r Note that it might be that M1f and M2f return the same memory vector.
		s_hat = M2f(s_hat)  # compute M2\shat
		
		t     = A(s_hat)    # compute A*shat
		
		omega = vecdot(t,r) / vecdot(t,t) ;
		
		BLAS.axpy!(N,omega,s_hat,1,x,1); 	# x = x + s_hat*omega
		BLAS.axpy!(N,-omega,t,1,r,1); 		# r = r - omega * t
		
		rnorm = vecnorm( r ) / bnrm2
		resvec[iter+1] = rnorm
		if out==2
			println(@sprintf("%3d\t%1.2e",iter,resvec[iter+1]))
		end
		if ( rnorm <= tol )
			flag = 0; break
		end
		if norm(omega) < 1e-16; flag = -4; break; end
		
	end
	if out>=0
		if flag==-1
			println(@sprintf("block bicgstb iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",
																								maxIter,resvec[iter],tol))
		elseif flag==-2
			println(@sprintf("block bicgstb: rho equal to zero at iteration %d. Returned residual has norm %1.2e.", iter,resvec[iter+1]))
		elseif flag==-4
			println(@sprintf("block bicgstb : omega < 1e-16"))
		elseif out>=1
			println(@sprintf("block bcgstb achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[iter+1]))
		end
	end
	return x, flag,resvec[iter+1],iter,resvec[1:iter+1]
end
