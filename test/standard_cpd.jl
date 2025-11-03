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

    opt_A = als_optimize(A, cp_A; alg = ITensorCPD.direct(), check, verbose);
    @test norm(ITensorCPD.reconstruct(opt_A) - A) / norm(A) < 1e-5

    svd_opt_A = als_optimize(A, cp_A; alg = ITensorCPD.TargetDecomp(), check);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(svd_opt_A)) /
          norm(ITensorCPD.reconstruct(opt_A)) < 1e-5

    check = ITensorCPD.FitCheck(1e-6, 20, norm(A))

    ## This method uses the interpolative squared to precondition the problem.
    alg = ITensorCPD.QRPivProjected(800)
    als = ITensorCPD.compute_als(A, cp_A; alg, check);
    
    als = ITensorCPD.update_samples(inds(cp_A), als, 900; reshuffle = true);
    @test ITensorCPD.stop(als.mttkrp_alg) == 900
    @test ITensorCPD.start(als.mttkrp_alg) == 1
    ITensorCPD.optimize(cp_A, als; verbose = true);
    
    int_opt_A =
       als_optimize(A, cp_A; alg = ITensorCPD.QRPivProjected(500), check, verbose=true, shuffle_pivots=true, trunc_tol=0.001);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(int_opt_A)) /
         norm(ITensorCPD.reconstruct(opt_A)) < 1e-2

    alg = ITensorCPD.SEQRCSPivProjected(1, 800, (1,2,3),(100,100,100))
    als = ITensorCPD.compute_als(A, cp_A; alg, check);
    
    als = ITensorCPD.update_samples(inds(cp_A), als, 600; reshuffle = true);
    @test ITensorCPD.stop(als.mttkrp_alg) == 600
    @test ITensorCPD.start(als.mttkrp_alg) == 1
    ITensorCPD.optimize(cp_A, als; verbose = true);

    int_opt_A =
        als_optimize(A, cp_A; alg = ITensorCPD.SEQRCSPivProjected((1,), (800,), (1,2,3),(100,100,100)),
        check, verbose, shuffle_pivots=false);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(int_opt_A)) /
          norm(ITensorCPD.reconstruct(opt_A)) < 1e-1

    direct_inversion_opt_A = als_optimize(A, cp_A; alg = ITensorCPD.InvKRP(), check, verbose);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(direct_inversion_opt_A)) /
          norm(ITensorCPD.reconstruct(opt_A)) < 1e-2

    
    ## This tests to see if we can interpolate a known low rank tensor
    A = ITensorCPD.reconstruct(random_CPD(A, 20))
    check = ITensorCPD.CPDiffCheck(1e-5, 100)

    cp_A = random_CPD(A, 10)
    opt_A = ITensorCPD.als_optimize(A, cp_A; check, verbose);
    exact_error = norm(A - ITensorCPD.reconstruct(opt_A)) / norm(A)
    int_opt_A =
        als_optimize(A, cp_A; alg = ITensorCPD.QRPivProjected((1,1,1), (1200, 800, 600)), check);
    @test abs(exact_error - norm(A - ITensorCPD.reconstruct(int_opt_A)) / norm(A)) / exact_error < 0.1

    ### Test for Leverage score sampling CPD 
    a,b,c = Index.((12,13,3))
    T = random_itensor(elt, a,b,c)

    cpdT = random_CPD(T, 5)
    T = reconstruct(cpdT)

    cpd = random_CPD(T, 5)
    alg = ITensorCPD.LevScoreSampled()
    check = ITensorCPD.CPDiffCheck(1e-5, 10)
    cpd_opt = ITensorCPD.als_optimize(T, cpd; alg, check, verbose);

    alg = ITensorCPD.LevScoreSampled(100)
    check=ITensorCPD.FitCheck(1e-3, 5, norm(T))
    cpd_opt = ITensorCPD.als_optimize(T, cpd; alg, check, verbose);
    @test norm(reconstruct(cpd_opt) - T) / norm(T) < 0.1

    ### Test for Leverage score sampling CPD 
    alg = ITensorCPD.LevScoreSampled((50, 50, 500))
    min_val = 1
    for i in 1:3
        cpd_opt = ITensorCPD.als_optimize(T, cpd; alg, check, verbose);
        val = norm(reconstruct(cpd_opt) - T) / norm(T) 
        min_val = val < min_val ? val : val
    end
     @test min_val < 0.1
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

    check = ITensorCPD.FitCheck(1e-6, 20, norm(A))


    ## This method uses the interpolative squared to precondition the problem.
    int_opt_A =
       als_optimize(A, cp_A; alg = ITensorCPD.QRPivProjected(), check, verbose);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(int_opt_A)) /
         norm(ITensorCPD.reconstruct(opt_A)) < 1e-2

    int_opt_A =
        als_optimize(A, cp_A; alg = ITensorCPD.SEQRCSPivProjected((1,1,1), (20*40, 20*40, 20*30), (1,2,3),(100,100,100)),check, verbose);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(int_opt_A)) /
          norm(ITensorCPD.reconstruct(opt_A)) < 1e-1

    int_opt_A =
        als_optimize(A, cp_A; alg = ITensorCPD.SEQRCSPivProjected((1,1,1), (20*40, 20*40, 20*30), (2),(100,)),check, verbose);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(int_opt_A)) /
          norm(ITensorCPD.reconstruct(opt_A)) < 1e-1

    direct_inversion_opt_A = als_optimize(A, cp_A; alg = ITensorCPD.InvKRP(), check, verbose);
    @test norm(ITensorCPD.reconstruct(opt_A) - ITensorCPD.reconstruct(direct_inversion_opt_A)) /
          norm(ITensorCPD.reconstruct(opt_A)) < 1e-2

    
    ## This tests to see if we can interpolate a known low rank tensor
    A = ITensorCPD.reconstruct(random_CPD(A, 20))

    cp_A = random_CPD(A, 10)
    check=ITensorCPD.FitCheck(1e-3, 5, norm(A))
    opt_A = ITensorCPD.als_optimize(A, cp_A; check, verbose=true);
    exact_error = norm(A - ITensorCPD.reconstruct(opt_A)) / norm(A)
    int_opt_A =
        als_optimize(A, cp_A; alg = ITensorCPD.QRPivProjected(1200), check, verbose=true);
    @test abs(exact_error - norm(A - ITensorCPD.reconstruct(int_opt_A)) / norm(A)) / exact_error < 0.1

    ### Test for Leverage score sampling CPD 
    a,b,c = Index.((12,13,3))
    T = random_itensor(elt, a,b,c)

    cpdT = random_CPD(T, 5)
    T = reconstruct(cpdT)

    cpd = random_CPD(T, 5)
    alg = ITensorCPD.LevScoreSampled()
    check = ITensorCPD.CPDiffCheck(1e-5, 10)
    cpd_opt = ITensorCPD.als_optimize(T, cpd; alg, check, verbose);

    alg = ITensorCPD.LevScoreSampled(100)
    cpd_opt = ITensorCPD.als_optimize(T, cpd; alg, check, verbose);
    @test norm(reconstruct(cpd_opt) - T) / norm(T) < 0.1
end


@testset "Build CPD to error threshold, elt=$elt" for elt in [Float64, ComplexF64]
    verbose = false
    i, j, k = Index.((20, 30, 40))
    r = Index(400, "CP_rank")
    A = random_itensor(elt, i, j, k)

    opt_A = ITensorCPD.decompose(A, 1e-3, 400; check=ITensorCPD.FitCheck(1e-4, 100, norm(A)), start_rank = 200, rank_step = 200, verbose=false);
    
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-3
end