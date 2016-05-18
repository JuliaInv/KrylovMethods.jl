using KrylovMethods
using Base.Test

m = 20
n = 10
A = randn(m,n);

Aold  = copy(A)
Qc,Rc = cgs(A)
@test all(Aold.==A) # make sure input is unchanged
Qt,Rt = cgs!(A)
@test all(Qt.==A) # check that input IS overwritten
@test all(Qt.==Qc)
@test norm(Aold-Qc*Rc)/norm(A) < 1e-15
@test norm(Qc'*Qc-eye(n)) < 1e-15

Aold  = copy(A)
Qm,Rm = mgs(A)
@test all(Aold.==A) # make sure input is unchanged
Qt,Rt = mgs!(A)
@test all(Qt.==A) # check that input IS overwritten
@test all(Qt.==Qm)
@test norm(Aold-Qm*Rm)/norm(A) < 1e-15
@test norm(Qm'*Qm-eye(n)) < 1e-15

