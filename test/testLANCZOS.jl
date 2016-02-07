using KrylovMethods
using Base.Test
using MatrixDepot


println("=== Testing Lanczos helpers ===")
symOrth = KrylovMethods.symOrtho

v     = ([0;randn()],[randn();0],randn(2))
for k=1:3
	println("Givens rotation for v=[$(v[k][1]),$(v[k][2])]")
	c,s,r = symOrth(v[k][1],v[k][2])
	@test abs(c.^2 + s.^2 - 1) < 1e-14
	Q = [c s; -s c]
	w = [norm(v[k]);0]
	@test norm(Q*v[k]-w) < 1e-13
end

println("=== Testing Lanczos Bidiagonalization ===")
A = sprandn(10,10,.4) + speye(10)
p = randn(10)
k = 9

U,B,V = lanczosBidiag(A,p,k)

@test norm(full(U'*A*V-B))/norm(full(A)) < 1e-15


println("=== Testing Lanczos Tridiagonalization ===")

# test stopping for low-rank matrix
A = randn(100,10)
H = A*A'
T,V = lanczosTridiag(H,ones(100),100,doReorth=true)
@test size(T,1)==11
@test size(V,2)==11

A   = matrixdepot("rosser",4)
Aff  = x -> A*x
T1,V1 = lanczosTridiag(A,ones(size(A,1)),size(A,1),tol=1e-10,doReorth=true)
T2,V2 = lanczosTridiag(Aff,ones(size(A,1)),size(A,1),tol=1e-10,doReorth=false)
sT1  = svd(full(T1))[2]
sT2  = svd(full(T2))[2]
sA   = svd(full(A))[2]

@test abs(sT1[1]-sA[1])/sA[1] < 1e-4
@test abs(sT2[1]-sA[1])/sA[1] < 1e-4
B = V1'*V1
@test norm(B - diagm(diag(B))) < 1e-10
@test norm(diag(B)- ones(size(B,1))) < 1e-10
println("=== Lanczos: All tests passed. ===")
