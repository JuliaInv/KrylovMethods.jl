using KrylovMethods
using Base.Test

m = 20
n = 10
A = randn(m,n);

Qc,Rc = cgs(A)
@test norm(A-Qc*Rc)/norm(A) < 1e-15
@test norm(Qc'*Qc-eye(n)) < 1e-15

Qm,Rm = mgs(A)
@test norm(A-Qm*Rm)/norm(A) < 1e-15
@test norm(Qm'*Qm-eye(n)) < 1e-15

