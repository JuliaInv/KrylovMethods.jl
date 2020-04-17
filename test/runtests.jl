using KrylovMethods
using Test
using LinearOperators
using SparseArrays
using LinearAlgebra
using Printf

include("getDivGrad.jl")

@testset "KrylovMethods" begin

include("testBICGSTB.jl")
include("testBlockBiCGSTB.jl")
include("testCG.jl")
include("testBlockCG.jl")
include("testCGLS.jl")
include("testGS.jl")
include("testGMRES.jl")
include("testFGMRES.jl")
include("testBlockFGMRES.jl")
include("testLANCZOS.jl")
include("testSSOR.jl")
include("testLSQR.jl")
include("testMINRES.jl")
# include("testMatrixDepot.jl")

end
