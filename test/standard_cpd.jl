@testset "Standard CPD, elt=$elt" for elt in [Float64, ComplexF64]
  i, j, k = Index.((20, 30, 40))
  r = Index(400, "CP_rank")
  A = random_itensor(elt, i, j, k)
  ## Calling decompose
  opt_A = ITensorCPD.decompose(A, r);
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-7

  @test_throws TypeError ITensorCPD.decompose(A, 400; solver=A)

  check = ITensorCPD.FitCheck(1e-15, 100, norm(A))
  opt_A = ITensorCPD.decompose(A, 400; check);

  ## Build a random guess
  cp_A = random_CPD(A, r)
  
  ## Optimize with no inputs
  opt_A = als_optimize(A, cp_A; check, verbose=true);
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-5

  ## Optimize with one input
  opt_A = als_optimize(A, cp_A; alg=ITensorCPD.KRP())
  @test norm(reconstruct(opt_A) - A) / norm(A) < 5e-7

  opt_A = als_optimize(A, cp_A; alg=ITensorCPD.KRP(), check=ITensorCPD.NoCheck(10))
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-1

  opt_A = als_optimize(A, cp_A;  alg=ITensorCPD.KRP(), check)
  @test norm(ITensorCPD.reconstruct(opt_A) - A) / norm(A) < 1e-5

  opt_A = als_optimize(A, cp_A; alg=ITensorCPD.direct())
  @test norm(reconstruct(opt_A) - A) / norm(A) < 5e-7

  opt_A = als_optimize(A, cp_A; alg=ITensorCPD.direct(), check)
  @test norm(ITensorCPD.reconstruct(opt_A) - A) / norm(A) < 1e-5
end

@testset "Standard CPD, elt=$elt" for elt in [Float32, ComplexF32]
  i, j, k = Index.((20, 30, 40))
  r = Index(4, "CP_rank")
  A = random_itensor(elt, i, j, k)
  ## Calling decompose

  cp = ITensorCPD.random_CPD(A, r);
  A = reconstruct(cp)

  check = ITensorCPD.FitCheck(1e-10, 100, norm(A))
  ops = ITensorCPD.decompose(A, r; verbose = true, check);
  opt_A = ITensorCPD.decompose(A, r);
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-3

  r = Index(400, "CP_rank")
  @test_throws TypeError ITensorCPD.decompose(A, r; solver=A)

  check = ITensorCPD.FitCheck(1e-8, 100, norm(A))
  opt_A = ITensorCPD.decompose(A, r; check, verbose = true);

  ## Build a random guess
  cp_A = random_CPD(A, r)
  
  ## Optimize with no inputs
  opt_A = als_optimize(A, cp_A; check, verbose=true);
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-5

  ## Optimize with one input
  opt_A = als_optimize(A, cp_A; alg=ITensorCPD.KRP())
  @test norm(reconstruct(opt_A) - A) / norm(A) < 5e-5

  opt_A = als_optimize(A, cp_A; alg=ITensorCPD.KRP(), check=ITensorCPD.NoCheck(10))
  @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-1

  opt_A = als_optimize(A, cp_A;  alg=ITensorCPD.KRP(), check)
  @test norm(ITensorCPD.reconstruct(opt_A) - A) / norm(A) < 1e-5

  opt_A = als_optimize(A, cp_A; alg=ITensorCPD.direct())
  @test norm(reconstruct(opt_A) - A) / norm(A) < 5e-5

  opt_A = als_optimize(A, cp_A; alg=ITensorCPD.direct(), check)
  @test norm(ITensorCPD.reconstruct(opt_A) - A) / norm(A) < 1e-5
end