using ITensors
using ITensorCPD: ITensorCPD, als_optimize
using LinearAlgebra
using Random

using ITensorNetworks: ITensorNetworks, ITensorNetwork, IndsNetwork, commoninds, delta_network, edges, src, dst, degree, insert_linkinds
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

function ising_network(
  eltype::Type,
  s::IndsNetwork,
  beta::Number;
  h::Number = 0.0,
  szverts = nothing,
)
  s = insert_linkinds(s; link_space = 2)
  # s = insert_missing_internal_inds(s, edges(s); internal_inds_space=2)
  tn = delta_network(eltype, s)
  if (szverts != nothing)
      for v in szverts
          tn[v] = diagITensor(eltype[1, -1], inds(tn[v]))
      end
  end
  for edge in edges(tn)
      v1 = src(edge)
      v2 = dst(edge)
      i = commoninds(tn[v1], tn[v2])[1]
      deg_v1 = degree(tn, v1)
      deg_v2 = degree(tn, v2)
      f11 = exp(beta * (1 + h / deg_v1 + h / deg_v2))
      f12 = exp(beta * (-1 + h / deg_v1 - h / deg_v2))
      f21 = exp(beta * (-1 - h / deg_v1 + h / deg_v2))
      f22 = exp(beta * (1 - h / deg_v1 - h / deg_v2))
      q = eltype[f11 f12; f21 f22]
      w, V = eigen(q)
      w = map(sqrt, w)
      sqrt_q = V * ITensors.Diagonal(w) * inv(V)
      t = itensor(sqrt_q, i, i')
      tn[v1] = tn[v1] * t
      tn[v1] = noprime!(tn[v1])
      t = itensor(sqrt_q, i', i)
      tn[v2] = tn[v2] * t
      tn[v2] = noprime!(tn[v2])
  end
  return tn
end

function replace_inner_w_prime_loop(tn)
  ntn = deepcopy(tn)
  for i = 1:(length(tn)-1)
      cis = inds(tn[i])
      is = commoninds(tn[i], tn[i+1])
      nis = [i ∈ is ? i' : i for i in cis]
      replaceinds!(ntn[i], cis, nis)
      cis = inds(tn[i+1])
      nis = [i ∈ is ? i' : i for i in cis]
      replaceinds!(ntn[i+1], cis, nis)
  end

  i = length(tn)
  cis = inds(tn[i])
  is = commoninds(tn[i], tn[1])
  nis = [i ∈ is ? i' : i for i in cis]
  replaceinds!(ntn[i], cis, nis)
  cis = inds(tn[1])
  nis = [i ∈ is ? i' : i for i in cis]
  replaceinds!(ntn[1], cis, nis)
  return ntn
end

function norm_of_loop(s1::ITensorNetwork)
  sising = s1.data_graph.vertex_data.values
  sisingp = replace_inner_w_prime_loop(sising)

  sqrs = sising[1] * sisingp[1]
  for i = 2:length(sising)
      sqrs = sqrs * sising[i] * sisingp[i]
  end
  return sqrt(sqrs[])
end

function ring_inds(start::Int, nx::Int, ny::Int)
  inds = Vector{Tuple{Int,Int}}()
  for y = start:(ny-start+1)
      push!(inds, (start, y))
  end
  for x = start+1:(nx-start+1)
      push!(inds, (x, ny - start + 1))
  end
  for y = (ny-start+1):-1:start
      push!(inds, (nx - start + 1, y))
  end
  for x = (nx-start+1):-1:start+1
      push!(inds, (x, start))
  end
  return Tuple(unique!(inds))
end

### Set up A 5 x 5 Grid ####
nx = 5
ny = 5
s = IndsNetwork(named_grid((nx, ny)); link_space = 2)

### Fill the 5 x 5 grid with the Ising partition function ### 
beta = 0.4
elt = Float64
tn = ising_network(elt, s, beta)

nranks_vals = Vector{Vector{Float64}}()
kranks = [2,3,4,10,20,50,100]

ring_subtn = subgraph(tn, ring_inds(2,5,5))
for known_ranks in kranks[1:3]
  r = Index(known_ranks, "CPD")
  # rcpd_edge = NDTensors.data(ITensorCPD.reconstruct(ITensorCPD.random_CPD(tn[2,2], r)))
  for v in ITensorNetworks.vertices(tn)
    # d = NDTensors.data(tn[v])
    # tn[v] = itensor(view(rcpd_edge, 1:length(d)), inds(tn[v]))
    tn[v] = ITensorCPD.reconstruct(ITensorCPD.random_CPD(tn[v], r))
  end

  vals = Vector{Float64}()
  for rank in 10:5:100
    # fit = ITensorCPD.FitCheck(1e-8, 1000, norm(tn[3,4]))
    # r = Index(rank, "CP_rank")
    # f = 0
    # for i in 1:5
    #   initial_guess = ITensorCPD.random_CPD(tn[2,2], r; rng = Random.MersenneTwister(rand(Int32)));
    #   @time cpopt = ITensorCPD.als_optimize(initial_guess, r, fit; verbose=false);
    #   curr_f = ITensorCPD.fit(fit)
    #   f = curr_f > f ? curr_f : f
    # end
    ring_subtn = subgraph(tn, ((2,2),(2,3),(3,3),(3,2)))
    #ring_subtn = subgraph(tn, ring_inds(2,5,5))
    n_ringsubtn = norm_of_loop(ring_subtn)
    fit = ITensorCPD.FitCheck(1e-8, 1000, n_ringsubtn)
    r = Index(Int(round(rank)), "CP_rank")
    @show r
    f = 0;
    for i in 1:1
      initial_guess = ITensorCPD.random_CPD_ITensorNetwork(ring_subtn, r; rng = Random.MersenneTwister(rand(Int32)));
      @time cpopt = ITensorCPD.als_optimize(initial_guess, r, fit;verbose=true);
      curr_f = ITensorCPD.fit(fit)
      f = curr_f > f ? curr_f : f
    end
    push!(vals, f)
  end
  push!(nranks_vals, vals)
end

using Plots
p = plot()
for r in 1:3
  p = plot!([Int(round(x)) for x in 10:5:100], 1 .- nranks_vals[r], label="Known rank of $(kranks[r])")
end
p
plot!(title="Full inner ring", ylabel="L2 relative error", xlabel="rank", yrange=[0,0.01])

savefig("/mnt/home/kpierce/.julia/dev/ITensorCPD/experiments/experiment_plots/known_ranks/full_ring.pdf")
