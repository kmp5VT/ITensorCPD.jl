using ITensorNetworks
using ITensorNetworks: random_tensornetwork
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors
include("./util.jl")

# @testset "Known rank Network" for elt in (Float32, Float64)
#   nx = 3
#   grid = named_grid((nx,2))
#   tn1 = random_tensornetwork(grid; link_space = 1)
#   tn2 = random_tensornetwork(grid; link_space = 1)
#   tn = tn1 + tn2

#   subtn = subgraph(tn, ((1,1),(2,1),(3,1)))    
#   s = subtn.data_graph.vertex_data.values
#   sp = replace_inner_w_prime_loop(s)

#   sqrs = s[1] * sp[1]
#   for i = 2:length(sp)
#       sqrs = sqrs * s[i] * sp[i]
#   end

#   check = ITensorCPD.FitCheck(1e-10, 1000, sqrt(sqrs[]))

#   ITensorCPD.decompose(subtn, Index(2,"rank"); check, verbose=false);
#   check.final_fit ≈ 1.0

#   nx = ny = 5
#   grid = named_grid((nx,ny))
#   tn1 = random_tensornetwork(grid; link_space = 1)
#   tn2 = random_tensornetwork(grid; link_space = 1)

#   tn = tn1 + tn2

#   subtn = subgraph(tn, ((2,2),(2,3),(2,4), 
#                           (3,4), (4,4),
#                           (4,3), (4,2),
#                           (3,2))) 

#   s = subtn.data_graph.vertex_data.values
#   sp = replace_inner_w_prime_loop(s)

#   sqrs = s[1] * sp[1]
#   for i = 2:length(sp)
#       sqrs = sqrs * s[i] * sp[i]
#   end
#   check = ITensorCPD.FitCheck(1e-20, 1000, sqrt(sqrs[]))

#   using Random
#   rng = MersenneTwister(3)
#   bestfit = 0;
#   opt = nothing
#   for i in 1:10
#       opt = ITensorCPD.decompose(subtn, Index(2,"rank"); verbose=false, rng);
#       fit = 1.0 - (norm(reconstruct(opt) - contract(subtn)) / norm(contract(subtn)))
#       bestfit = fit > bestfit ? fit : bestfit
#   end
#   @test bestfit ≈ 1.0
# end

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
    cpopt = ITensorCPD.als_optimize(s1, ITensorCPD.random_CPD(s1, r); check, verbose = false);
    #1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract([s1...])) / sqrs[]
    @test isapprox(
        check.final_fit,
        1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract([s1...])) / sqrs[],
        rtol = 1e-3,
    )
end

# TODO this is still broken so open an issue about complex tensor networks. Single tensors are able to be complex.
# @testset "itensor_networks" for elt in (ComplexF32, ComplexF64)
#   nx = 3
#   ny = 3
#   s = IndsNetwork(named_grid((nx, ny)); link_space = 2)

#   beta = 1
#   tn = ising_network(elt, s, beta)

#   r = Index(10, "CP_rank")
#   s1 = subgraph(tn, ((1, 1), (1, 2),))

#   sising = s1.data_graph.vertex_data.values
#   ## TODO make this with ITensorNetworks
#   sisingp = replace_inner_w_prime_loop(sising)

#   sqrs = sising[1] * sisingp[1]
#   for i = 2:length(sising)
#       sqrs = sqrs * sising[i] * sisingp[i]
#   end

#   check = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
#   cpopt = ITensorCPD.decompose(s1,  Index(4, "CP_rank"); check, verbose=true);
#   #cpopt = ITensorCPD.als_optimize(s1, ITensorCPD.random_CPD(s1, r); check, verbose = true)
#   1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract(s1)) / norm(contract(s1))
#   @test isapprox(
#       check.final_fit,
#       1.0 - norm(ITensorCPD.reconstruct(cpopt) - contract(s1)) / norm(contract(s1));
#       rtol = 1e-3,
#   )
# end
