using ITensorNetworks
using ITensorNetworks: random_tensornetwork
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors
using Random
include("./util.jl")

@testset "Known rank Network: eltype=:$(elt)" for elt in
                                                 (Float32, Float64, ComplexF32, ComplexF64)
    nx = 3
    grid = named_grid((nx, 3))
    tn1 = random_tensornetwork(elt, grid; link_space = 1)
    tn2 = random_tensornetwork(elt, grid; link_space = 1)
    tn = tn1 + tn2

    subtn = subgraph(tn, ((2, 1), (2, 2), (2, 3)))
    s = subtn.data_graph.vertex_data.values
    sp = replace_inner_w_prime_loop(s)

    sqrs = s[1] * dag(sp[1])
    for i = 2:length(sp)
        sqrs = sqrs * s[i] * dag(sp[i])
    end

    check = ITensorCPD.FitCheck(1e-10, 100, real(sqrt(sqrs[])))

    while check.final_fit < 0.9
        rng = Random.seed!(Random.RandomDevice())
        guess = ITensorCPD.random_CPD(subtn, 2; rng)
        cpd = ITensorCPD.als_optimize(subtn, guess; check, verbose = true);
    end
    @test 1 - check.final_fit < 0.1

    ## TODO this is a test to see if physical problems are well interpolated with the 
    ## new algorithms
    # rng = Random.seed!(Random.RandomDevice());
    # subtn = contract(subtn);
    # guess = ITensorCPD.random_CPD(subtn, 2; rng);

    # alg = ITensorCPD.QRPivProjected(1, 2)
    # cpd = ITensorCPD.als_optimize(subtn, guess; check, verbose = false, alg);
    # diff = subtn - reconstruct(cpd)
    # @show norm(diff) / norm(subtn)

    nx = ny = 5
    grid = named_grid((nx, ny))
    tn1 = random_tensornetwork(elt, grid; link_space = 1)
    tn2 = random_tensornetwork(elt, grid; link_space = 1)

    tn = tn1 + tn2

    subtn = subgraph(tn, ((2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (3, 2)))

    s = subtn.data_graph.vertex_data.values
    sp = replace_inner_w_prime_loop(s)

    sqrs = s[1] * dag(sp[1])
    for i = 2:length(sp)
        sqrs = sqrs * s[i] * dag(sp[i])
    end
    check = ITensorCPD.FitCheck(1e-20, 1000, real(sqrt(sqrs[])))

    using Random
    rng = MersenneTwister(3)
    bestfit = 0;
    opt = nothing
    for i = 1:3
        opt = ITensorCPD.decompose(subtn, Index(2, "rank"); check, verbose = true, rng);
        fit = check.final_fit
        bestfit = fit > bestfit ? fit : bestfit
    end
    @test 1 - bestfit < 0.01
end

@testset "itensor_networks" for elt in (Float32, Float64, ComplexF32, ComplexF64)
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

    sqrs = sising[1] * dag(sisingp[1])
    for i = 2:length(sising)
        sqrs = sqrs * sising[i] * dag(sisingp[i])
    end


    check = ITensorCPD.FitCheck(1e-3, 6, real(sqrt(sqrs[])))
    cpopt = ITensorCPD.als_optimize(s1, ITensorCPD.random_CPD(s1, r); check, verbose = false);
    #1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract([s1...])) / sqrs[]
    @test isapprox(
        check.final_fit,
        1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract([s1...])) / sqrs[],
        rtol = 1e-3,
    )
end
