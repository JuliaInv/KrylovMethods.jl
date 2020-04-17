
@testset "GS" begin

m = 20
n = 10
A = randn(m,n);

Aold  = copy(A)
Qc,Rc = cgs(A)
@test all(Aold.==A) # make sure input is unchanged
Qt,Rt = cgs!(A)
@test all(Qt.==A) # check that input IS overwritten
@test maximum(abs.(Qt-Qc)) < 1e-15
@test norm(Aold-Qc*Rc,Inf)/norm(A,Inf) < 1e-14
@test norm(Qc'*Qc-I,Inf) < 1e-14

Aold  = copy(A)
Qm,Rm = mgs(A)
@test all(Aold.==A) # make sure input is unchanged
Qt,Rt = mgs!(A)
@test all(Qt.==A) # check that input IS overwritten
@test maximum(abs.(Qt-Qm)) < 1e-15
@test norm(Aold-Qm*Rm,Inf)/norm(A) < 1e-14
@test norm(Qm'*Qm-I,Inf) < 1e-14

end
