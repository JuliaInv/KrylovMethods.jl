@testset "MINRES" begin

A = sprandn(10,10,0.1)
A = A'*A + 4*I
Af = (v) -> A*v
rhs = randn(10)
rhst = copy(rhs)
# test stop by maxIter
x0,f0, = minres(A,rhs,maxIter=1,rtol=1e-12,gtol=1e-12,btol=1e-20,out=1,storeInterm=true)
@test f0 == -1

x1,f1, = minres(A,rhs,maxIter=10,rtol=1e-12,gtol=1e-12,btol=1e-20,out=2)
@test all(rhs.==rhst)
x2,f2, = minres(Af,rhs,x=x0[:,1],maxIter=10,rtol=1e-12,gtol=1e-12,btol=1e-20,out=2)
@test all(rhs.==rhst)

@test f1==0
@test (norm(A*x1-rhs)/norm(rhs)) < 1e-11
@test (norm(A*x2-rhs)/norm(rhs)) < 1e-11
@test norm(x1-x2)/norm(x1) < 1e-12
println("test stop by condlim")
# A = matrixdepot("fiedler",50)
# rhs = randn(50)
# x,f, = minres(A,rhs,maxIter=50,rtol=1e-12,gtol=1e-12,btol=1e-20,condlim=200)
# @test f==-3

println("test stop by beta")
A = randn(20,4)
A = A*A'
rhs = randn(20)
x,f, = minres(A,rhs,maxIter=20,rtol=1e-20,gtol=1e-20,btol=1e-10,condlim=1e20)
@test f==-2

println("test if anything is done for zero right hand side")
A = randn(20,4)
mv = 0
Af = (x) -> ( mv+=1; return Af*x)
rhs = zeros(20)
x, = minres(Af,rhs,maxIter=20,rtol=1e-20,gtol=1e-20,btol=1e-10,condlim=1e20)
@test all(x.==0)
@test mv==0
end
