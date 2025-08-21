@testset "Standard CPD, elt=$elt" for elt in [Float64, ComplexF64]
    verbose = false
    i, j, k = Index.((20, 30, 40))
    r = Index(400, "CP_rank")
    A = random_itensor(elt, i, j, k)
    ## Calling decompose
    opt_A = ITensorCPD.decompose(A, r);
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-7

    @test_throws TypeError ITensorCPD.decompose(A, 400; solver = A)

    check = ITensorCPD.FitCheck(1e-6, 100, norm(A))
    opt_A = ITensorCPD.decompose(A, 400; check, verbose);
    @test norm(A - reconstruct(opt_A)) / norm(A) < 1e-5

    ## Build a random guess
    cp_A = random_CPD(A, r)

    ## Optimize with no inputs
    opt_A = als_optimize(A, cp_A; check, verbose)
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-5

    ## Optimize with one input
    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.KRP());
    @test norm(reconstruct(opt_A) - A) / norm(A) < 5e-7

    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.KRP(), check = ITensorCPD.NoCheck(10))
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-1

    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.KRP(), check)
    @test norm(ITensorCPD.reconstruct(opt_A) - A) / norm(A) < 1e-5

    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.direct())
    @test norm(reconstruct(opt_A) - A) / norm(A) < 5e-7

    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.direct(), check, verbose=true);
    @test norm(ITensorCPD.reconstruct(opt_A) - A) / norm(A) < 1e-5

    svd_opt_A = als_optimize(A, cp_A; alg = ITensorCPD.TargetDecomp(), check);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(svd_opt_A)) /
          norm(ITensorCPD.reconstruct(opt_A)) < 1e-5

    check = ITensorCPD.NoCheck(35)

    ## Test broken 
    # int_opt_A =
    #     als_optimize(A, cp_A; alg = ITensorCPD.InterpolateTarget(), check);
    # @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(int_opt_A)) /
    #       norm(ITensorCPD.reconstruct(opt_A)) < 1e-5

    ## This method uses the interpolative squared to precondition the problem.
    int_opt_A =
        als_optimize(A, cp_A; alg = ITensorCPD.DoubleInterp(), check);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(int_opt_A)) /
          norm(ITensorCPD.reconstruct(opt_A)) < 1e-2

    direct_inversion_opt_A = als_optimize(A, cp_A; alg = ITensorCPD.InvKRP(), check);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(direct_inversion_opt_A)) /
          norm(ITensorCPD.reconstruct(opt_A)) < 1e-2
end

@testset "Standard CPD, elt=$elt" for elt in [Float32, ComplexF32]
    verbose = false
    i, j, k = Index.((20, 30, 40))
    r = Index(4, "CP_rank")
    A = random_itensor(elt, i, j, k)
    ## Calling decompose

    cp = ITensorCPD.random_CPD(A, r)
    A = reconstruct(cp)

    r = Index(400, "CP_rank")
    check = ITensorCPD.FitCheck(1e-6, 100, norm(A))
    opt_A = ITensorCPD.decompose(A, r; verbose, check)
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-3

    @test_throws TypeError ITensorCPD.decompose(A, r; solver = A)

    check = ITensorCPD.FitCheck(1e-6, 100, norm(A))
    # opt_A = ITensorCPD.decompose(A, r; check, verbose)

    ## Build a random guess
    cp_A = random_CPD(A, r)

    ## Optimize with no inputs
    opt_A = als_optimize(A, cp_A; check, verbose)
    @test norm(reconstruct(opt_A) - A) / norm(A) < 5e-5

    ## Optimize with one input
    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.KRP())
    @test norm(reconstruct(opt_A) - A) / norm(A) < 5e-5

    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.KRP(), check = ITensorCPD.NoCheck(10))
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-1

    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.KRP(), check)
    @test norm(ITensorCPD.reconstruct(opt_A) - A) / norm(A) < 5e-5

    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.direct())
    @test norm(reconstruct(opt_A) - A) / norm(A) < 5e-5

    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.direct(), check)
    @test norm(ITensorCPD.reconstruct(opt_A) - A) / norm(A) < 5e-5
end


@testset "Build CPD to error threshold, elt=$elt" for elt in [Float64, ComplexF64]
    verbose = false
    i, j, k = Index.((20, 30, 40))
    r = Index(400, "CP_rank")
    A = random_itensor(elt, i, j, k)

    opt_A = ITensorCPD.decompose(A, 1e-3, 400; check=ITensorCPD.FitCheck(1e-4, 100, norm(A)), start_rank = 200, rank_step = 200, verbose=true);
    
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-3

end