using LinearAlgebra
using ITensorCPD:SEQRCS, sparse_sign_matrix

### testing the generation of sparse matrix
@testset "S-Hashing matrix generation" begin
    m = 200      
    n = 10000       
    s = 8        
    l = 1200
    omega = sparse_sign_matrix(l, n, s) 
    @test size(omega) == (l, n)
    @test all(sum(!iszero, omega[:, j]) == s for j in 1:n)
    @test all(v == 0.0 || v == 1/sqrt(s) || v == -1/sqrt(s) for v in omega)

    A = randn(n,m)
    S=svd(A)
    act_singular = S.S
    A_sk = omega*A
    S_sk = svd(A_sk)
    sk_singular = S_sk.S
    @test all(0.5<=v<=1.5 for v in sk_singular./act_singular)
end

### testing the performance of SE-QRCS
@testset "SE-QRCS factorization test" begin
    m,n,k=50,10000,40
    A=randn(m,n)
    F = qr(A, ColumnNorm())
    Q_act,R_act,p_act=F.Q,F.R,F.p
    Q,R,p = SEQRCS(A,2500,1,40,40)
    error_act = norm(A[:,p_act]-Q_act[:,1:k]*R_act[1:k,:],2)/norm(A,2)
    error = norm(A[:,p]-Q[:,1:k]*R[1:k,:],2)/norm(A,2)
    @test (abs(error - error_act) <= 1e-2)
end