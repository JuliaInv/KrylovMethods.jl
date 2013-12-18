module SimpleIterativeSolvers
	include("cg.jl")
	include("cgls.jl")
	include("bicgstb.jl")
	include("gmres.jl")	
	include("lanczosBidiag.jl")
	include("sor.jl")	
end