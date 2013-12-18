export gmres

function gmres(A,b::Vector,restrt::Int; tol::Real=1e-2,maxIter::Int=100,M=x->x,x::Vector=[],out::Int=0)
# x,flag,err,iter,resvec = gmres(A,b,restrt,tol=1e-2,maxIter=100,M=1,x=[],out=0)
#
# Generalized Minimal residual ( GMRESm ) method with restarts applied to A*x = b.
#
# Input:
#
#	A       - matrix or function computing A*x
#	b       - right hand side vector
#	restrt  - number of iterations between restarts
#	tol     - error tolerance
#	maxIter - maximum number of iterations
#	M       - preconditioner, either matrix or function computing M\x
#	x       - starting guess
#	out     - flag for output (0 : only errors, 1 : final status, 2: error at each iteration)
#
# Output:
#
#	x       - approximate solution
#	flag    - exit flag (  0 : desired tolerance achieved,
#	                      -1 : maxIter reached without converging)
#	err     - norm of relative residual, i.e., norm(A*x-b)/norm(b)
#	iter    - number of iterations
#	resvec  - norm of relative residual at each iteration

	Af =  isa(A,Function) ? A : x->A*x
	Mf =  isa(M,Function) ? M : x->M\x
	
	# initialization
	n  = length(b)
	if isempty(x)
		x = zeros(n)
		r = Mf(b)
	else
		r = Mf(b-Af(x))
	end
	
	bnrm2 = norm(b)
	if bnrm2 == 0.0; bnrm2 = 1.0; end
	
	err = norm( r ) / bnrm2
	if err < tol; return x, err; end
	
	V     = zeros(n,restrt+1)
	H     = zeros(restrt+1,restrt)  
	cs    = zeros(restrt)  
	sn    = zeros(restrt)
	e1    = zeros(n)
	e1[1] = 1.0
	
	if iseltype(b,Complex) || iseltype(r,Complex) || iseltype(A,Complex)
		e1 = complex(e1)
		b  = complex(b)
		r  = complex(r)
		x  = complex(x)
		V  = complex(V)
		H  = complex(H)
		cs = complex(cs)
		sn = complex(sn)
	end
	
	resvec = zeros((1+restrt)*maxIter)
	if out==2
		println(@sprintf("=== gmres ===\n%4s\t%7s\n","iter","relres"))
	end
	
	iter = 0
	flag = -1
	cnt  = 1
	for iter = 1:maxIter
		V[:,1] = r / norm( r )
		s      = norm( r )*e1;  
		
		if out==2;; print(@sprintf("%3d\t", iter));end
		
		for i = 1:restrt
			w = Af(V[:,i])
			w = Mf(w)
			
			for k = 1:i # basis using Gram-Schmidt
				H[k,i] = dot(w,V[:,k])
				w     -= H[k,i]*V[:,k]
			end
			H[i+1,i] = norm( w )
			V[:,i+1] = w / H[i+1,i]
			
			for k = 1:i-1 # apply Givens rotation
				temp     =  cs[k]*H[k,i] + sn[k]*H[k+1,i]
				H[k+1,i] = -sn[k]*H[k,i] + cs[k]*H[k+1,i]
				H[k,i]   = temp
			end
			
			# Approximate residual norm
			cs[i],sn[i] = rotmat( H[i,i], H[i+1,i] )
			s[i+1] = -sn[i]*s[i]
			s[i]   = cs[i]*s[i]
			H[i,i] = cs[i]*H[i,i] + sn[i]*H[i+1,i]
			H[i+1,i] = 0.0
			err  = abs(s[i+1]) / bnrm2
			
			if out==2; print(@sprintf("%1.1e ", err));end
	
			resvec[cnt] = err
			
			if err <= tol
				y  = H[1:i,1:i] \ s[1:i]
				x += V[:,1:i]*y
				if out==2; print("\n"); end
				flag = 0; break
			end
			cnt = cnt+1
		end
		if  err <= tol
			flag = 0
			break
		end
		y  = H[1:restrt,1:restrt]\s[1:restrt]
		x += V[:,1:restrt]*y
		
		r = b - Af(x)
		r = Mf(r)
		
		s[restrt+1] = norm(r)
		resvec[cnt] = abs(s[restrt+1]) / bnrm2
		
		if out==2; print(@sprintf("\t %1.1e\n", err)); end
		if err <= tol
		   flag = 0; break
		end
	end
	
	if flag==-1
		println(@sprintf("gmres iterated maxIter (=%d) times without achieving the desired tolerance.",maxIter))
	elseif out>=1
		println(@sprintf("gmres achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[cnt]))
	end
	return x,flag,resvec[cnt],iter,resvec[1:cnt]
end

function rotmat( a, b )
# c,s = rotmat(a,b)
#  Givens rotation matrix parameters for a and b.

	if b==0.0
		c = 1.0
		s = 0.0
	elseif abs(b) > abs(a)
		temp = a / b
		s    = 1.0/sqrt( 1.0 + temp^2 )
		c    = temp * s
	else
		temp = b/a
		c    = 1.0/sqrt( 1.0 + temp^2 )
		s    = temp * c
	end
	return c, s
end
