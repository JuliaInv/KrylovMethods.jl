using KrylovMethods
using Base.Test
using MatrixDepot
using LinearOperators

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
include("testLANCZOS.jl")
include("testSSOR.jl")
include("testLSQR.jl")
include("testMINRES.jl")
include("testMatrixDepot.jl")

end