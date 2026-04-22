using LinearAlgebra,Random
using ITensorCPD:SEQRCS, sparse_sign_matrix
using SparseArrays: sparse

### testing the generation of sparse matrix
@testset "S-Hashing matrix generation" begin
    m = 10      
    n = 10000       
    s = 8        
    l = 1200
    rows = Vector{Int32}(undef, n * s)
    vals = Vector{Float64}(undef, n * s)
    omega = sparse_sign_matrix(l, n, s, rows, vals;omega=true)
    
    @test size(omega) == (l, n)
    @test all(sum(!iszero, omega[:, j]) == s for j in 1:n)
    @test length(rows) == (n * s)
    @test all(v == 0.0 || v == 1/sqrt(s) || v == -1/sqrt(s) for v in vals)

    A = randn(n,m)
    S=svd(A)
    act_singular = S.S
    A_sk_old = omega*A
    A_sk = ITensorCPD.sketched_matricization(itensor(A', Index.(size(A'))), 1, l, rows, vals, s)
    @test norm(A_sk_old - A_sk') ≈ 0
    S_sk = svd(A_sk')
    sk_singular = S_sk.S
    @test all(0.5<=v<=1.5 for v in sk_singular./act_singular)
end

## testing the performance of SE-QRCS
@testset "SE-QRCS factorization test" begin
    k=40
    A=randn(50,10000)
    m, n = Index.((50, 10000))
    A_tensor = itensor(A,m,n)
    F = qr(A, ColumnNorm())
    Q_act,R_act,p_act=F.Q,F.R,F.p
    Q,R,p = SEQRCS(A_tensor,1,m,2500,1,40)
    error_act = norm(A[:,p_act]-Q_act[:,1:k]*R_act[1:k,:],2)/norm(A,2)
    error = norm(A[:,p]-Q[:,1:k]*R[1:k,:],2)/norm(A,2)
    @test (abs(error - error_act) <= 1e-2)

    Q,R,p = SEQRCS(A_tensor,1,m,2500,1,40; use_omega=true)
    error_act = norm(A[:,p_act]-Q_act[:,1:k]*R_act[1:k,:],2)/norm(A,2)
    error = norm(A[:,p]-Q[:,1:k]*R[1:k,:],2)/norm(A,2)
    @test (abs(error - error_act) <= 1e-2)

    ## Test the SE-QRCS of a KRP structured tensor.
    cpd = ITensorCPD.random_CPD(zeros(50, 100, 100), 50)
    cprank = ITensorCPD.cp_rank(cpd)
    krp = ITensorCPD.had_contract(cpd[2],cpd[3], cprank)

    krpmat = itensor(array(krp, inds(krp)[[3,1,2]]), m,n)
    Q,R,p = SEQRCS(krpmat, 1, m, 2500, 1, 40)

    error = norm(array(krpmat)[:,p]-Q[:,1:k]*R[1:k,:],2)/norm(array(krpmat),2)
    Qf,Rf,pf = SEQRCS([cpd[2], cpd[3]], cprank, 2500, 1, 40);
    error_fact = norm(array(krpmat)[:,pf]-Qf[:,1:k]*Rf[1:k,:],2)/norm(array(krpmat),2)

    @test abs(error - error_fact) ≤ 1e-2
end