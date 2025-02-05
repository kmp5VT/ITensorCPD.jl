using ITensorNetworks
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors

using ITensorCPD:
  ITensorCPD,
  als_optimize,
  direct,
  random_CPD,
  random_CPD_square_network,
  row_norm,
  reconstruct,
  had_contract

  include("$(@__DIR__)/helpers.jl")

  function contract_loop(r1, tn, i, vs_centre)
    core = subgraph(tn, ((3,3),(3,4)))
    s1 = subgraph(tn, ring_inds(2, 5, 6))
    s2 = subgraph(tn, ring_inds(1, 5, 6))
  
  
    fit = ITensorCPD.FitCheck(1e-4,1000,norm_of_loop(s1))
    cp_guess = ITensorCPD.random_CPD_ITensorNetwork(s1, r1)
    cpopt = ITensorCPD.als_optimize(cp_guess, r1, fit);
    outer, inner = ITensorCPD.tn_cp_contract(s2, cpopt);
    val = (itensor(array(ITensorCPD.had_contract(outer.data_graph.vertex_data.values, r1)) .* array(cpopt[]), r1)  * ITensorCPD.had_contract([inner..., vs_centre...], r1))[]
    return vals[i] = val
  end
  
  ##############################################
  nx = 5
  ny = 6

  s = IndsNetwork(named_grid((nx, ny)); link_space=2);
  vals = [0.0, 0.0]
cp_szsz = Vector{Vector{Float64}}([])
cp_szsz_old = copy(cp_szsz)
ranks = [1, 6, 15]
cp_guess = nothing
h = 0
vs_centre = nothing
szverts=[(3, 3), (3, 4)]
for rank in 1:length(ranks)
  push!(cp_szsz, Vector{Float64}(undef, 0))
  r1 = Index(ranks[rank], "CP_rank")
  r2 = Index(ranks[rank], "CP_rank")
  for beta in [1.0 - i for i in 0:0.01:1]
   for i in 1:2
     if i == 1
        tn = ising_network(Float64, s, beta; h)
      else
        tn = ising_network(Float64, s, beta; h, szverts)
      end
      #if isnothing(env)
      vs_centre = [tn[x] for x in szverts]
      val = contract_loop(r1, tn, i, [tn[x] for x in szverts])
      #contract_loops(r1, r2, tn, i; old_contract = false)
      # contract_loop_exact_core(r1, r2, tn, i)
    end
    cp = vals[2] / vals[1]
    push!(cp_szsz[rank], cp)
  end
end

println()

include("refs_and_plots.jl")


#########################################################################
# plots
using Plots
plot(betas, theor[end:-1:1]; label="Infinite Lattice", s=:solid, )
plot!(betas, full_szsz; label="Exact Contraction", s=:dash,)
plot!(betas, bp_szsz; label="BP Contraction", s=:dot,)
plot!(betas, cp_szsz[1]; label="CP rank 1", s=:auto,)
plot!(betas, cp_szsz[2]; label="CP rank 6", s=:auto, )
plot!(betas, cp_szsz[3]; label="CP rank 15", s =:auto,)
@show plot!(;
  xlabel="Inverse Temparature",
  ylabel="SZ Correlation",
  legend=:bottomright,
  title="CPD contraction of 2D network, 6x5 grid",
)
savefig("$(@__DIR__)/../../experiment_plots/cp_ising_5x6.pdf")








# ranks  = [1, 5, 10, 20, 50]
# for rank in 1:length(ranks)
#   r1 = Index(ranks[rank], "CP rank")
# tn = ising_network(Float64, s, beta; h)
# s1 = subgraph(tn, ring_inds(2, 5, 6))
# fit = ITensorCPD.FitCheck(1e-4,1000,norm_of_loop(s1))
# cp_guess = ITensorCPD.random_CPD_ITensorNetwork(s1, r1)
# cpopt = ITensorCPD.als_optimize(cp_guess, r1, fit);
# println()
# end


# vals_beta_1 = [0.28736104101687954, 0.8774065244829014, 0.8831709759007664 , 0.8942831975670831, 0.9237155659346836]
# vals_beta_04 = [0.09819469320441554, 0.33165378975952875, 0.42075509970486724, 0.5173803302686404, 0.6952755933787372]
# using Plots
# plot(ranks, vals_beta_1; label="β=1.0, h = 0")
# plot!(ranks, vals_beta_04; label="β=0.4, h = 0")
# plot!(;title="CP rank vs L2 fit", xlabel="CP rank", ylabel="L2 fit", ylim=[0,1])
# savefig("$(@__DIR__)/l2_fit.pdf")
##########################################################
    #==
      s1_c = contract(s1);
      inds(s1_c)
      core = tn[3,3] * tn[3,4]
      u,s,v = svd(s1_c, commoninds(s1_c ,core))
      using Plots
      plot(data(s))
  
      fit = ITensorCPD.FitCheck(1e-4, 1000, norm(s1_c))
      r1 = Index(1, "CP_rank")
      cpopt = ITensorCPD.als_optimize(ITensorCPD.random_CPD_ITensorNetwork(s1, r1), r1, fit);
      plot!(sort(data(cpopt[]); rev=true); label="rank 1", ms=:square, )
      s1_c = ITensorCPD.reconstruct(cpopt)
      u, sp1, c = svd(s1_c,  commoninds(s1_c ,core))
      sp1 = cpopt[]
      plot(((data(s) .- data(sp1)) ./ data(s))[1:1], label="CP rank 1", m=:circle)
  
      r1 = Index(5, "CP_rank")
      cpopt = ITensorCPD.als_optimize(ITensorCPD.random_CPD_ITensorNetwork(s1, r1), r1, fit);
      plot!(sort(data(cpopt[]); rev=true); label="rank 5", ms=:square, )
      s1_c = ITensorCPD.reconstruct(cpopt)
      u, sp5, c = svd(s1_c,  commoninds(s1_c ,core))
      sp5 = cpopt[]
      plot!(((data(s)[1:5] .- data(sp5)) ./ data(s)[1:5]), label="CP rank 5", m=:circle)
  
      using Random
      r1 = Index(10, "CP_rank")
      cpopt = ITensorCPD.als_optimize(ITensorCPD.random_CPD_ITensorNetwork(s1, r1; rng=MersenneTwister(rand(Int64))), r1, fit);
      plot!(sort(data(cpopt[]); rev=true); label="rank 10", ms=:square, )
      s1_c = ITensorCPD.reconstruct(cpopt)
      u, sp10, c = svd(s1_c,  commoninds(s1_c ,core))
      sp10 = cpopt[]
      plot!(((data(s)[1:10] .- data(sp10)) ./ data(s)[1:10]), label="CP rank 10", m=:circle)
  
  
      r1 = Index(20, "CP_rank")
      cpopt = ITensorCPD.als_optimize(ITensorCPD.random_CPD_ITensorNetwork(s1, r1; rng=MersenneTwister(rand(Int64))), r1, fit);
      s1_c = ITensorCPD.reconstruct(cpopt)
      u, sp20, c = svd(s1_c,  commoninds(s1_c ,core))
      sp20 = cpopt[]
      plot!(((data(s)[1:20] .- data(sp20)) ./ data(s)[1:20]), label="CP rank 20", m=:circle)
  
  
      r1 = Index(50, "CP_rank")
      cpopt = ITensorCPD.als_optimize(ITensorCPD.random_CPD_ITensorNetwork(s1, r1; rng=MersenneTwister(rand(Int64))), r1, fit);
      plot!(sort(data(cpopt[]); rev=true); label="rank 50", ms=:square, )
      s1_c = ITensorCPD.reconstruct(cpopt)
      u, sp50, c = svd(s1_c,  commoninds(s1_c ,core))
      sp50 = cpopt[]
      plot!(((data(s)[1:50] .- data(sp50)) ./ data(s)[1:50])[1:30], label="CP rank 50", m=:circle)
  
      plot!(;title="Error in shell singular values versus CP lambda values", ylabel="L2 error", xlabel="SVD number")
      savefig("Shell_svd_cpd_error.pdf")
    ==#