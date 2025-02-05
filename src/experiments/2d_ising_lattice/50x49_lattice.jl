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

function contract_loops(r::Int, tn, nx, ny, i, vs_centre)
  # grab the furthest most edge
  sout = subgraph(tn, ring_inds(1, nx, ny))
  # And the next inside edge
  sin = subgraph(tn, ring_inds(2, nx, ny))
  
  fit = ITensorCPD.FitCheck(1e-4,10, norm_of_loop(sin))
  # Compute the CPD of the inside edge
  cpr = Index(r, "CP_rank_2")
  cp_guess = ITensorCPD.random_CPD_ITensorNetwork(sin, cpr)
  cpopt_in = ITensorCPD.als_optimize(cp_guess, cpr; maxiters=10);

  # contract the legs going out with the outside graph and also keep the legs goin inside
  outer, to_core = ITensorCPD.tn_cp_contract(sout, cpopt_in);
  # contract all of the tensors on the outer edge (there are now just r chains so this is cheap)
  outside_edge = itensor(array(ITensorCPD.had_contract(outer.data_graph.vertex_data.values, cpr)) .* array(cpopt_in[]), cpr) 
  # outside edge is simply a vector now.

  # loop over all of the next inner shells except for the center shell
  center = maximum([nx,ny]) ÷ 2
  for sg in 3:(center - 1)
    # grab the next inner subgraph
    sin = subgraph(tn, ring_inds(sg, nx, ny))

    # compute the CPD of this next shell
    fit = ITensorCPD.FitCheck(1e-4,10, norm_of_loop(sin))
    cpr = Index(r, "cp_rank_$(sg)")
    cp_guess = ITensorCPD.random_CPD_ITensorNetwork(sin, cpr)
    cpopt_in = ITensorCPD.als_optimize(cp_guess, cpr; maxiters=10);

    # contract the to_core
    outer, _, to_core = ITensorCPD.cp_cp_contract(to_core, cpopt_in.factors);
    outside_edge = itensor(array(outside_edge * outer) .* array(cpopt_in[]), cpr)
  end

    core = ITensorCPD.had_contract([to_core...], cpr)
    return outside_edge * core
    # val = (outside_edge * core)[]
    # return vals[i] = val
end
  
  ##############################################
  nx = 49
  ny = 50
  h = 0

  s = IndsNetwork(named_grid((nx, ny)); link_space=2);
  vals = [0.0, 0.0]
cp_szsz = Vector{Vector{Float64}}([])
cp_szsz_old = copy(cp_szsz)
ranks = [15]
cp_guess = nothing
szverts=[(25, 25), (25, 26)]
h = 0
for rank in 1:length(ranks)
 push!(cp_szsz, Vector{Float64}(undef, 0))
  for beta in [1.0 - i for i in 0:0.01:1]
   for i in 1:2
      if i == 1
        tn = ising_network(Float64, s, beta; h)
      else
        tn = ising_network(Float64, s, beta; h, szverts)
      end
      #if isnothing(env)
      vs_centre = [tn[x] for x in szverts]
      core = contract_loops(ranks[rank], tn, nx, ny, i, vs_centre)
      vals[i] = (contract([core, vs_centre...])[])
      #contract_loops(r1, r2, tn, i; old_contract = false)
      # contract_loop_exact_core(r1, r2, tn, i)
    end
    cp = vals[2] / vals[1]
    push!(cp_szsz[rank], cp)
  end
end

using Plots
cp_szsz[1] = cp_szsz[1][102:end]
plot(betas, abs.(full_szsz .- cp_szsz[1]); label="CP rank 1")
plot!(betas, abs.(full_szsz .- cp_szsz[2]); label="CP rank 6")
plot!(betas, abs.(full_szsz .- cp_szsz[3]); label="CP rank 15")
plot!(betas, abs.(full_szsz .- bp_szsz); label="BP")
plot!(; xlabel="Inverse Temparature", ylabel="SZ Correlation error")
savefig("../CP_ising_2site_14x13.pdf")