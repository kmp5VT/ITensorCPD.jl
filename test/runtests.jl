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
using ITensors: Index, ITensor, array, contract, dim, norm, random_itensor

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

## Complex optimization still does not work
@testset "Standard CPD, elt=$elt" for elt in [Float32, Float64]
    ## Working here
    i, j, k = Index.((20, 30, 40))
    r = Index(400, "CP_rank")
    A = random_itensor(elt, i, j, k)
    cp_A = random_CPD(A, r)

    opt_A = als_optimize(cp_A, r; maxiters = 100)
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-7

    check = ITensorCPD.FitCheck(1e-15, 100, norm(A))
    opt_A = als_optimize(cp_A, r, check)
    @test norm(reconstruct(opt_A) - A) / norm(A) ≤ 1.0 - ITensorCPD.fit(check)

    cp_A = random_CPD(A, r; algorithm = direct())
    opt_A = als_optimize(cp_A, r; maxiters = 100)
    @test norm(reconstruct(opt_A) - A) / norm(A) < 1e-7

    check = ITensorCPD.FitCheck(1e-15, 100, norm(A))
    cp_A = random_CPD(A, r; algorithm = direct())
    opt_A = als_optimize(cp_A, r, check)
    @test norm(reconstruct(opt_A) - A) / norm(A) ≤ 1.0 - ITensorCPD.fit(check)
end

using ITensorNetworks
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors
include("util.jl")

@testset "itensor_networks" for elt in (Float32, Float64)
    nx = 3
    ny = 3
    s = IndsNetwork(named_grid((nx, ny)); link_space = 2)

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

    fit = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
    cpopt = ITensorCPD.als_optimize(ITensorCPD.random_CPD_ITensorNetwork(s1, r), r, fit)
    1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract(s1)) / norm(contract(s1))
    @test isapprox(
        fit.final_fit,
        1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract(s1)) / norm(contract(s1));
        rtol = 1e-3,
    )
end
