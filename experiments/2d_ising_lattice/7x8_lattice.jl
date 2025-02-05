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

#==
  function potts_network(
    eltype::Type, s::IndsNetwork, beta::Number; h::Number=0.0, szverts=nothing
  )
    s = insert_linkinds(s; link_space=3)
    # s = insert_missing_internal_inds(s, edges(s); internal_inds_space=2)
    tn = delta_network(eltype, s)
    if (szverts != nothing)
      for v in szverts
        tn[v] = diagITensor(eltype[1, -1], inds(tn[v]))
      end
    end
    # for edge in edges(tn)
    #   v1 = src(edge)
    #   v2 = dst(edge)
    #   i = commoninds(tn[v1], tn[v2])[1]
    #   deg_v1 = degree(tn, v1)
    #   deg_v2 = degree(tn, v2)
    #   f11 = exp(beta * (1 + h / deg_v1 + h / deg_v2))
    #   f12 = exp(beta * (-1 + h / deg_v1 - h / deg_v2))
    #   f21 = exp(beta * (-1 - h / deg_v1 + h / deg_v2))
    #   f22 = exp(beta * (1 - h / deg_v1 - h / deg_v2))
    #   q = eltype[f11 f12; f21 f22]
    #   w, V = eigen(q)
    #   w = map(sqrt, w)
    #   sqrt_q = V * ITensors.Diagonal(w) * inv(V)
    #   t = itensor(sqrt_q, i, i')
    #   tn[v1] = tn[v1] * t
    #   tn[v1] = noprime!(tn[v1])
    #   t = itensor(sqrt_q, i', i)
    #   tn[v2] = tn[v2] * t
    #   tn[v2] = noprime!(tn[v2])
    # end
    return tn
  end
  ptn = potts_network(Float64, s, 1.0)
==#

function contract_loops(r1, r2, tn, i, vs_centre)
  souter = subgraph(tn, ring_inds(1,7,8))
  s1 = subgraph(tn, ring_inds(2, 7, 8))

  fit = ITensorCPD.FitCheck(1e-3, 100, norm_of_loop(s1))
  cp = ITensorCPD.random_CPD_ITensorNetwork(s1, r1)
  @time cpopt = ITensorCPD.als_optimize(cp, r1, fit)

  ## contract s1 with outer layer   
  sout, factors = ITensorCPD.tn_cp_contract(souter, cpopt)
  core = cpopt.factors
  sout = ITensorCPD.had_contract(sout.data_graph.vertex_data.values, r1)
  v = itensor(array(sout) .* array(cpopt[]), r1)

  s2 = subgraph(tn, ring_inds(3,7,8))

  fit = ITensorCPD.FitCheck(1e-3, 10, norm_of_loop(s2))
  cp = ITensorCPD.random_CPD_ITensorNetwork(s2, r2)
  @time cpopt = ITensorCPD.als_optimize(cp, r2, fit)

  had, es, core = ITensorCPD.cp_cp_contract(core, cpopt.factors)

  is = ind.(core, 2)
  #env = ITensor(Float64, is);
  l = had * v
  val = 0
  for i in 1:dim(r2)
    #env += cpopt[][i] * l[i] * contract([itensor(array(x)[i,:], ind(x,2)) for x in core])
    val += (cpopt[][i] * (l[i] * contract([
      itensor(array(x)[i, :], ind(x, 2)) for x in core
    ])) * vs_centre[1] * vs_centre[2])[]
  end
  return vals[i] = val
end

function contract_loop_exact_core(r1, r2, tn, i; check_svd::Bool=false)
  s1 = subgraph(
    tn,
    (
      (2, 2),
      (2, 3),
      (2, 4),
      (2, 5),
      (2, 6),
      (2, 7),
      (3, 7),
      (4, 7),
      (5, 7),
      (6, 7),
      (6, 6),
      (6, 5),
      (6, 4),
      (6, 3),
      (6, 2),
      (5, 2),
      (4, 2),
      (3, 2),
    ),
  )

  sising = s1.data_graph.vertex_data.values
  sisingp = replace_inner_w_prime_loop(sising)

  sqrs = sising[1] * sisingp[1]
  for i in 2:length(sising)
    sqrs = sqrs * sising[i] * sisingp[i]
  end
  sqrt(sqrs[])

  fit = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
  cp = ITensorCPD.random_CPD_square_network(sising, r1)
  @time cpopt = ITensorCPD.als_optimize(cp, r1, fit)
  ## contract s1 with outer layer   
  core = [
    cpopt[4],
    cpopt[6],
    cpopt[8],
    cpopt[10],
    cpopt[13],
    cpopt[15],
    cpopt[17],
    cpopt[21],
    cpopt[23],
    cpopt[25],
    cpopt[27],
    cpopt[32],
    cpopt[34],
    cpopt[36],
  ]

  es = [
    cpopt[2] * tn[1, 2] * tn[1, 1]
    cpopt[3] * tn[1, 3]
    cpopt[5] * tn[1, 4]
    cpopt[7] * tn[1, 5]
    cpopt[9] * tn[1, 6]
    cpopt[11] * tn[1, 7] * tn[1, 8]
    cpopt[12] * tn[2, 8]
    cpopt[14] * tn[3, 8]
    cpopt[16] * tn[4, 8]
    cpopt[18] * tn[5, 8]
    cpopt[20] * tn[6, 8] * tn[7, 8]
    cpopt[19] * tn[7, 7]
    cpopt[22] * tn[7, 6]
    cpopt[24] * tn[7, 5]
    cpopt[26] * tn[7, 4]
    cpopt[28] * tn[7, 3]
    cpopt[30] * tn[7, 2] * tn[7, 1]
    cpopt[29] * tn[6, 1]
    cpopt[31] * tn[5, 1]
    cpopt[33] * tn[4, 1]
    cpopt[35] * tn[3, 1]
    cpopt[1] * tn[2, 1]
  ]

  v = Vector{Float64}(undef, dim(r1))
  for i in 1:dim(r1)
    v[i] = contract([itensor(array(x)[i, :, :], inds(x)[2:end]) for x in es])[] * cpopt[][i]
  end
  v = itensor(v, r1)

  next_layer = [
    core[1] * tn[3, 3] # * core[14]
    core[2] * tn[3, 4]
    core[3] * tn[3, 5]
    core[4] * tn[3, 6] # * core[5]
    core[6] * tn[4, 6]
    core[7] * tn[5, 6] # * core[8]
    core[9] * tn[5, 5]
    core[10] * tn[5, 4]
    core[11] * tn[5, 3] # * core[12]
    core[13] * tn[4, 3]
  ]

  result = 0
  for rank in 1:dim(r1)
    A = contract([
      itensor(array(x)[rank, :, :, :], inds(x)[2:end]) for x in next_layer[1:2]
    ])
    B = contract([
      itensor(array(x)[rank, :, :, :], inds(x)[2:end]) for x in next_layer[8:10]
    ])
    A = A * itensor(array(core[14])[rank, :], inds(core[14])[2])
    B = B * itensor(array(core[12])[rank, :], inds(core[12])[2])
    C1 = A * B * tn[4, 4]

    A = contract([
      itensor(array(x)[rank, :, :, :], inds(x)[2:end]) for x in next_layer[3:4]
    ])
    B = contract([
      itensor(array(x)[rank, :, :, :], inds(x)[2:end]) for x in next_layer[5:7]
    ])
    A = A * itensor(array(core[5])[rank, :], inds(core[5])[2])
    B = B * itensor(array(core[8])[rank, :], inds(core[8])[2])
    C2 = A * B * tn[4, 5]
    result += (C1 * C2)[] * v[rank]
  end
  return vals[i] = result
end

##############################################
nx = 7
ny = 8

s = IndsNetwork(named_grid((nx, ny)); link_space=2);

vals = [0.0, 0.0]
cp_szsz = Vector{Vector{Float64}}([])
cp_szsz_old = copy(cp_szsz)
ranks = [1, 6, 15]
szverts=[(4, 4), (4, 5)]
h = 0
vs_centre = nothing

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


      vs_centre = [tn[x] for x in szverts]
      contract_loops(r1, r2, tn, i, vs_centre)
      # contract_loop_exact_core(r1, r2, tn, i)
    end
    cp = vals[2] / vals[1]
    push!(cp_szsz[rank], cp)
  end
end



##################################################################
# refs

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
  title="CPD contraction of 2D network, 8x7 grid",
)
savefig("$(@__DIR__)/../../experiment_plots/cp_ising_7x8.pdf")

# plot(betas, abs.(full_szsz .- cp_szsz[1]); label="CP rank 1")
# plot!(betas, abs.(full_szsz .- cp_szsz[2]); label="CP rank 6")
# plot!(betas, abs.(full_szsz .- cp_szsz[3]); label="CP rank 10")
# plot!(betas, abs.(full_szsz .- cp_szsz[4]); label="CP rank 15")
# plot!(betas, abs.(full_szsz .- bp_szsz); label="BP")
# plot!(; xlabel="Inverse Temparature", ylabel="SZ Correlation error")
# savefig("../CP_ising_2site_error_exact_center.pdf")
