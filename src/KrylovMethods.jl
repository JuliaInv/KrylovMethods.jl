module KrylovMethods

	import Base.BLAS
	using Compat

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
