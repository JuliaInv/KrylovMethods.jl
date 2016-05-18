module KrylovMethods
	
import Base.BLAS

"""
function KrylovMethods.applyPrecond(M,P)
	
Apply preconditioner M, that is, compuyte Z = M\\P

Input:

  M       - preconditioner represented as Function, AbstractArray, or Vector
  P       - right hand side, Vector or Array

Output:

  Z       - M\\P, Vector or Array
"""
applyPrecond(M::Function,P) = M(P)
applyPrecond(M::AbstractArray,P::Vector) = M\P
applyPrecond(M::Vector,p::Vector) = M.\p
function applyPrecond(M::Vector,P::Array) 
	n,nrhs = size(P)
	V      = copy(P)
	for k=1:nrhs
		V[:,k] = M.\P[:,k]
	end
	return V
end


include("cg.jl")
include("blockCG.jl")
include("cgls.jl")
include("bicgstb.jl")
include("blockBiCGSTB.jl")
include("gmres.jl")	
include("lanczosBidiag.jl")
include("ssor.jl")	
include("lsqr.jl")	
include("lanczosTridiag.jl")
include("lanczos.jl")
include("minres.jl")
include("gs.jl")
end