using ITensorNetworks
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors
#include("$(@__DIR__)/../ITensorCPD.jl")
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

function contract_4_loops(r1, r2, r3, r4, tn, i, vs_centre)
    #core = subgraph(tn, ((3,3),(3,4)))
  s1 = subgraph(tn, ring_inds(5, 11, 12))
  s2 = subgraph(tn, ring_inds(4, 11, 12))
  s3 = subgraph(tn, ring_inds(3, 11, 12))
  s4 = subgraph(tn, ring_inds(2, 11, 12))
  s5 = subgraph(tn, ring_inds(1, 11, 12))
  
  
    #fit = ITensorCPD.FitCheck(1e-4,100, norm_of_loop(s3))
  cp_guess = ITensorCPD.random_CPD_ITensorNetwork(s4, r4)
  cpopt_s4 = ITensorCPD.als_optimize(cp_guess, r4; maxiters=10);
  outer, inner = ITensorCPD.tn_cp_contract(s5, cpopt_s4);
  t4 = itensor(array(ITensorCPD.had_contract(outer.data_graph.vertex_data.values, r4)) .* array(cpopt_s4[]), r4) 

  cp_guess = ITensorCPD.random_CPD_ITensorNetwork(s3, r3)
  cpopt_s3 = ITensorCPD.als_optimize(cp_guess, r3; maxiters=10);
  # contract cpopt with inner
  had, _, inner = ITensorCPD.cp_cp_contract(inner, cpopt_s3.factors)
  t3 = itensor(array(t4 * had) .* array(cpopt_s3[]), r3)

  #fit = ITensorCPD.FitCheck(1e-4, 100, norm_of_loop(s2))
  cp_guess = ITensorCPD.random_CPD_ITensorNetwork(s2, r2)
  cpopt_s2 = ITensorCPD.als_optimize(cp_guess, r2; maxiters=10);
  # contract cpopt with inner
  had, _, inner = ITensorCPD.cp_cp_contract(inner, cpopt_s2.factors)
  t2 = itensor(array(t3 * had) .* array(cpopt_s2[]), r2)

    #fit = ITensorCPD.FitCheck(1e-4, 100, norm_of_loop(s1))
    cp_guess = ITensorCPD.random_CPD_ITensorNetwork(s1, r1)
    cpopt_s1 = ITensorCPD.als_optimize(cp_guess, r1; maxiters=10)
    had, _, inner = ITensorCPD.cp_cp_contract(inner, cpopt_s1.factors)

    t1 = ITensorCPD.had_contract([inner..., vs_centre...], r1)
    val = (t2 * had * itensor(array(t1) .* array(cpopt_s1[]), r1))[]
    return vals[i] = val
  end
  
  ##############################################
  nx = 11
  ny = 12
  h = 0

  s = IndsNetwork(named_grid((nx, ny)); link_space=2);
  vals = [0.0, 0.0]
cp_szsz = Vector{Vector{Float64}}([])
cp_szsz_old = copy(cp_szsz)
ranks = [1, 6, 15]
cp_guess = nothing
szverts=[(6, 6), (6, 7)]
h = 0
for rank in 1:length(ranks)
 push!(cp_szsz, Vector{Float64}(undef, 0))
  r1 = Index(ranks[rank], "CP_rank1")
  r2 = Index(ranks[rank], "CP_rank2")
  r3 = Index(ranks[rank], "CP_rank3")
  r4 = Index(ranks[rank], "CP_rank4")
  for beta in [1.0 - i for i in 0:0.01:1]
   for i in 1:2
      if i == 1
        tn = ising_network(Float64, s, beta; h)
      else
        tn = ising_network(Float64, s, beta; h, szverts)
      end
      #if isnothing(env)
      vs_centre = [tn[x] for x in szverts]
      val = contract_4_loops(r1, r2, r3, r4, tn, i)
      #contract_loops(r1, r2, tn, i; old_contract = false)
      # contract_loop_exact_core(r1, r2, tn, i)
    end
    cp = vals[2] / vals[1]
    push!(cp_szsz[rank], cp)
  end
end

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
plot!(;
  xlabel="Inverse Temparature",
  ylabel="SZ Correlation",
  legend=:bottomright,
  title="CPD contraction of 2D network, 10x9 grid",
)
savefig("$(@__DIR__)/../../experiment_plots/cp_ising_9x10.pdf")