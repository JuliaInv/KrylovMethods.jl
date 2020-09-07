
export qmr

"""
 Quasi Minimal Residual (QMR) method applied to the complex symmetric system of
 linear equations Ax=b. The algorithm employed here was modified from Algorithm 8.1
 in the paper:
    Freund R W and Nachtigal N M. An implementation of the QMR method based on
      coupled two-term recurrences. SIAM J Sci., Comput., 1994, 15 (2): 313-337.

 - by Bo Han, May 2016, hanbo1735@163.com, China University of Geosciences.

Input:
    A       :: Function   - computing A*x.
    b       :: Vector     - right hand side vector.
    tol     :: Real       - error tolerance.
    maxIter :: Int        - maximum number of iterations.
    M       :: Function   - preconditioners, either matrices or functions computing M\\x.
    x0      :: Vector     - starting guess.
    out     :: Int        - flag for output (-1: no output, 0: only errors,
                                              1: final status, 2: relres at each iteration).

Output:
    x       :: Vector     - solution.
    flag    :: Int        - exit flag ( 0: desired tolerance achieved
                                       -1: maxIter reached without converging
                                       -2: delta equal to zero
                                       -3: epsilon equal to zero
                                       -9: right hand side was zero ).
    err     :: Real       - residual, i.e., norm(A*x-b)/norm(b).
    iter    :: Int        - number of iterations.
    resvec  :: Vector     - residual at each iteration.

"""
function qmr(A::SparseMatrixCSC{T,Int},b::Array{T,1}; kwargs...) where {T}
	Ax = zeros(T,size(A,2))                  # pre-allocate
	return qmr(x -> mul!(Ax,A,x,1.0,0.0),b;kwargs...) 
end

qmr(A,b::Vector;kwargs...) = qmr(x -> A*x ,b;kwargs...)

function qmr(A::Function, b::Vector; tol::Real=1e-6, maxIter::Int=100, M=x->copy(x),
             x::Vector=[],out::Int=0)
	println("WARNING: QMR: this (new) funtion does not seem to work very well at the moment. Need to check algorithm correctness. ")
    n  = length(b)
    if norm(b)==0; return zeros(eltype(b),n),-9,0,0,zeros(1); end
    Mf = isa(M,Function) ? M : x -> M\x

    # allocate vector: p,d,s
    p = zeros(eltype(b),n)
    d = zeros(eltype(b),n)
    s = zeros(eltype(b),n)

    # specify scalar
    c_n      =  one(eltype(b))
    eps      =  one(eltype(b))
    theta_n  =  zero(eltype(b))
    eta      =  -one(eltype(b))

    if isempty(x)
        x = zeros(eltype(b),n)
        r  = copy(b)
    else
		r  = b - A(x)
    end
	
	if eltype(b) <: Complex 
       x = complex(x)
    end

    resvec = zeros(maxIter+1)
    bnrm2 = norm( b )
    err   = norm( r ) / bnrm2; resvec[1] = err

    rho_np = norm( r )
    v = r / rho_np
    # v = BLAS.scal(n,one(eltype(r))/rho_np,r,1)

    if out==2
        println("=== qmr ===")
        println(@sprintf("%4s\t%7s","iter","relres"))
    end

    iter = 1
    flag = -1

    # start the iteration
    while iter <= maxIter
        c_nm     = c_n
        rho_n    = rho_np
        theta_nm = theta_n

        mv = Mf(v)
        #delta = At_mul_B(v, mv)    # delta = v.' * mv
        delta = transpose(v) * mv
        abs(delta) < 1e-40 && (flag=-2; break)

        # Update p:   p = M \ v - p * (rho*delta/eps)
        sctmp = -rho_n * delta / eps
        BLAS.scal!(n,sctmp,p,1)                 #  - p * (rho*delta/eps)
        BLAS.axpy!(n,one(eltype(p)),mv,1,p,1)   # p = p + mv

        ap   = A(p)
        # eps  = At_mul_B(p, ap)    # eps  = p.' * ap
        eps  = transpose(p) * ap
        abs(eps) < 1e-40 && (flag=-3; break)
        beta = eps / delta

        # Update v:   v = A * p - beta * v,   v=v/norm(v)
        BLAS.scal!(n,-beta,v,1)                  #  -beta * v
        BLAS.axpy!(n,one(eltype(v)),ap,1,v,1)    # v = v + ap
        rho_np = norm( v )
        BLAS.scal!(n,one(eltype(v))/rho_np,v,1)  # v = v / rho_np


        theta_n = rho_np^2 / c_nm / (real(beta)^2+imag(beta)^2)    # theta_n^2
        c_n     = 1.0 / (1.0 + theta_n)                            # c_n^2
        eta     = -eta * rho_n * c_n / (beta * c_nm)
        k2      = theta_nm * c_n

        # Update d:   d = eta * p + k2 * d
        BLAS.scal!(n,one(eltype(b))*k2,d,1)     # k2 * d
        BLAS.axpy!(n,eta,p,1,d,1)               # d = d + eta * p

        # Update x:  x = x + d
        BLAS.axpy!(n,one(eltype(d)),d,1,x,1)

        # Update s:  s = eta * ap + k2 * s
        BLAS.scal!(n,one(eltype(s))*k2,s,1)     # k2 * s
        BLAS.axpy!(n,eta,ap,1,s,1)              # s = s + eta * ap

        # Update r:  r = r - s
        BLAS.axpy!(n,-one(eltype(r)),s,1,r,1)

        err = norm( r ) / bnrm2
        resvec[iter+1] = err

        out==2 && println(@sprintf("%3d\t%1.2e",iter,resvec[iter+1]))
        iter += 1

        err <= tol && (flag=0; break)
    end  # iter = 1:maxIter
    iter = min(iter, maxIter)

    if out>=0
      if flag==-1
        println(@sprintf("qmr iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",
						 maxIter,resvec[iter],tol))
      elseif flag==-2
        println(@sprintf("qmr: delta equal to zero at iteration %d. Returned residual has norm %1.2e.", iter,resvec[iter]))
      elseif flag==-3
        println(@sprintf("qmr: epsilon equal to zero at iteration %d. Returned residual has norm %1.2e.", iter,resvec[iter]))
      elseif out>=1
	    println(@sprintf("qmr achieved desired tolerance at iteration %d. Residual norm is %1.2e.",iter,resvec[iter+1]))
	  end
    end


	return x, flag,resvec[iter+1],iter,resvec[1:iter+1]

end
