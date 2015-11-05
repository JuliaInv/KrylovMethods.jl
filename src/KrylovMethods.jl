module KrylovMethods
	
	import Base.BLAS
	
	include("cg.jl")
	include("cgls.jl")
	include("bicgstb.jl")
	include("gmres.jl")	
	include("lanczosBidiag.jl")
	include("ssor.jl")	
	include("lsqr.jl")	
	include("lanczosTridiag.jl")
	include("lanczos.jl")
	include("minres.jl")
end