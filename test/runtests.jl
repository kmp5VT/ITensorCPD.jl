using Test, Pkg
Pkg.develop(path = "$(@__DIR__)/../")

using ITensorCPD:
    ITensorCPD,
    als_optimize,
    direct,
    random_CPD,
    row_norm,
    reconstruct,
    had_contract
using ITensors: Index, ITensor, itensor, array, contract, dim, norm, random_itensor

@testset "Norm Row test, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
    i, j = Index.((20, 30))
    A = random_itensor(elt, i, j)
    Ainorm, lam = row_norm(A, i)
    for id = 1:dim(i)
        @test real(one(elt)) ≈ sum(array(Ainorm .^ 2)[:, id])
    end

    Ajnorm, lam = row_norm(A, j)
    for id = 1:dim(i)
        @test real(one(elt)) ≈ sum(array(Ajnorm .^ 2)[id, :])
    end

    Aijnorm, lam = row_norm(A, i, j)
    @test real(one(elt)) ≈ sum(array(Aijnorm .^ 2))
end

@testset "Hadamard contract algorithm, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
    i, j, k = Index.((20, 30, 40))
    r = Index(400, "CP_rank")
    A = random_itensor(elt, i, j, k)

    cp = random_CPD(A, r);

    Dhad = had_contract(cp[1],cp[2], r)

    Dex = ITensor(elt, r,i,j)
    for n in 1:400
        for l in 1:20
            for m in 1:30
                Dex[n,l,m] = cp[1][n,l] * cp[2][n,m]
            end
        end
    end

    @test norm(Dex - Dhad) / norm(Dex) < 1e-7
    
    v = Array(transpose(array(cp[2])))
    B = itensor(v, j,r)

    had_contract(cp[1], B, r)

    Dex = ITensor(elt, r,i,j)
    for n in 1:400
        for l in 1:20
            for m in 1:30
                Dex[n,l,m] = cp[1][n,l] * B[m,n]
            end
        end
    end

    @test norm(Dex - Dhad) / norm(Dex) < 1e-7
end

@testset "reconstruct, elt=$elt" for elt in [Float32, Float64, ComplexF32, ComplexF64]
    i, j = Index.((20, 30))
    r = Index(10, "CP_rank")
    A, B = random_itensor.(elt, ((r, i), (r, j)))
    λ = ITensor(randn(elt, dim(r)), r)
    exact = fill!(ITensor(elt, i, j), zero(elt))
    for R = 1:dim(r)
        for I = 1:dim(i)
            for J = 1:dim(j)
                exact[I, J] += λ[R] * A[R, I] * B[R, J]
            end
        end
    end
    recon = reconstruct([A, B], λ)
    @test 1.0 - norm(array(exact - recon)) / norm(exact) ≈ 1.0 rtol = eps(real(elt))

    k = Index.(40)
    A, B, C = random_itensor.(elt, ((r, i), (r, j), (r, k)))
    λ = ITensor(randn(elt, dim(r)), r)
    exact = fill!(ITensor(elt, i, j, k), zero(elt))
    for R = 1:dim(r)
        for I = 1:dim(i)
            for J = 1:dim(j)
                for K = 1:dim(k)
                    exact[I, J, K] += λ[R] * A[R, I] * B[R, J] * C[R, K]
                end
            end
        end
    end
    recon = reconstruct([A, B, C], λ)

    @test 1.0 - norm(array(exact - recon)) / norm(exact) ≈ 1.0 rtol = eps(real(elt))
end

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
    r = Index(3, "CP_rank")
    A = random_itensor(elt, i, j, k)
    ## Calling decompose

    cp = ITensorCPD.random_CPD(A, r);
    A = reconstruct(cp)

    check = ITensorCPD.FitCheck(1e-10, 100, norm(A))
    ops = ITensorCPD.decompose(A, r; verbose = true, check);
    array(ops[])
    array(cp[])
    opt_A = ITensorCPD.decompose(A, r);
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-3

    @test_throws TypeError ITensorCPD.decompose(A, 400; solver=A)

    check = ITensorCPD.FitCheck(1e-15, 100, norm(A))
    opt_A = ITensorCPD.decompose(A, 400; check, verbose = true);

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

using ITensorNetworks
using ITensorNetworks: random_tensornetwork
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors
include("util.jl")

#@testset "Known rank Network" for elt in (Float32, Float64)
    nx = 3
    grid = named_grid((nx,2))
    tn1 = random_tensornetwork(grid; link_space = 1)
    tn2 = random_tensornetwork(grid; link_space = 1)
    tn = tn1 + tn2

    subtn = subgraph(tn, ((1,1),(2,1),(3,1)))    
    s = subtn.data_graph.vertex_data.values
    sp = replace_inner_w_prime_loop(s)

    sqrs = s[1] * sp[1]
    for i = 2:length(sp)
        sqrs = sqrs * s[i] * sp[i]
    end

    check = ITensorCPD.FitCheck(1e-10, 1000, sqrt(sqrs[]))

    ITensorCPD.decompose(subtn, Index(2,"rank"); check, verbose=false);
    check.final_fit ≈ 1.0

    nx = ny = 5
    grid = named_grid((nx,ny))
    tn1 = random_tensornetwork(grid; link_space = 1)
    tn2 = random_tensornetwork(grid; link_space = 1)

    tn = tn1 + tn2

    subtn = subgraph(tn, ((2,2),(2,3),(2,4), 
                            (3,4), (4,4),
                            (4,3), (4,2),
                            (3,2))) 
                            
    s = subtn.data_graph.vertex_data.values
    sp = replace_inner_w_prime_loop(s)

    sqrs = s[1] * sp[1]
    for i = 2:length(sp)
        sqrs = sqrs * s[i] * sp[i]
    end
    check = ITensorCPD.FitCheck(1e-20, 1000, sqrt(sqrs[]))

    using Random
    rng = MersenneTwister(3)
    bestfit = 0;
    opt = nothing
    for i in 1:10
        opt = ITensorCPD.decompose(subtn, Index(2,"rank"); verbose=false, rng);
        fit = 1.0 - (norm(reconstruct(opt) - contract(subtn)) / norm(contract(subtn)))
        bestfit = fit > bestfit ? fit : bestfit
    end
    @test bestfit ≈ 1.0

    l = subgraph(tn, ((3,3),))
    n,c = ITensorCPD.tn_cp_contract(l, opt)
end

@testset "itensor_networks" for elt in (Float32, Float64)
    nx = 3
    ny = 3
    s = IndsNetwork(named_grid((nx, ny)); link_space = 2)

    beta = 1
    tn = ising_network(elt, s, beta)

    r = Index(10, "CP_rank")
    s1 = subgraph(tn, ((1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (2, 1)))

    sising = s1.data_graph.vertex_data.values
    ## TODO make this with ITensorNetworks
    sisingp = replace_inner_w_prime_loop(sising)

    sqrs = sising[1] * sisingp[1]
    for i = 2:length(sising)
        sqrs = sqrs * sising[i] * sisingp[i]
    end

    check = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
    cpopt = ITensorCPD.als_optimize(s1, ITensorCPD.random_CPD(s1, r); check, verbose = true)
    1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract(s1)) / norm(contract(s1))
    @test isapprox(
        check.final_fit,
        1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract(s1)) / norm(contract(s1));
        rtol = 1e-3,
    )
end
