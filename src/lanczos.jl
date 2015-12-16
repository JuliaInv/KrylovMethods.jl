
"""
alpha,beta,vk,vkm1 = LanczosStep!(A::Function,vk,vkm1,beta;sigma=0.0,tol=1e-10)

Performs a Lanczos step

Implementation is based on Table 2.1 in

Choi, S.-C. T. (2006). 
Iterative Methods for Singular Linear Equations and Least-squares Problems. 
Phd thesis, Stanford University.
"""
function LanczosStep!(A::Function,vk,vkm1,beta;sigma=0.0,tol=1e-10)
    n     = length(vk)
    
    pk    = A(vk)
    if sigma!=0.0
        BLAS.axpy!(n,-sigma,vk,1,pk,1) #pk   -= sigma*vk
    end
    
    alpha = BLAS.dot(n,vk,1,pk,1)  # alpha = dot(vk,pk)
    BLAS.axpy!(n,-alpha,vk,1,pk,1) # pk   -= alpha*vk
    BLAS.axpy!(n,-beta,vkm1,1,pk,1)# pk   -= beta*vkm1
    vkm1  = BLAS.blascopy!(n,vk,1,vkm1,1) # vkm1 = copy(vk)
    vk    = pk 
    beta  = BLAS.nrm2(n,vk,1)
    if beta>tol
         vk /= beta
    end
    return alpha, beta,vk,vkm1
end


"""
c,s,r = SymOrtho(a,b)

Computes a Givens rotation

Implementation is based on Table 2.9 in

Choi, S.-C. T. (2006). 
Iterative Methods for Singular Linear Equations and Least-squares Problems. 
Phd thesis, Stanford University.
"""
function symOrtho(a,b)
    c = 0.0; s = 0.0; r = 0.0
	if b==0
		s = 0.0
		r = abs(a)
		c = (a==0) ? c=1.0 : c = sign(a)
	elseif a == 0
		c = 0.0
		s = sign(b)
		r = abs(b)
	elseif abs(b) > abs(a)
		tau = a/b
		s   = sign(b)/sqrt(1+tau^2)
		c   = s*tau
		r   = b/s
	elseif abs(a) > abs(b)
		tau = b/a
		c   = sign(a)/sqrt(1+tau^2)
		s   = c*tau
		r   = a/c
	end
	return c,s,r
end