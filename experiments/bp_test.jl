using Pkg
using ITensorNetworks
using ITensorNetworks.NamedGraphs
using ITensorNetworks.NamedGraphs.GraphsExtensions: subgraph
using ITensorNetworks.NamedGraphs.NamedGraphGenerators: named_grid

using ITensorNetworks: IndsNetwork, delta_network, edges, src, dst, degree, insert_linkinds
using ITensors
include("$(@__DIR__)/../ITensorCPD.jl")
using .ITensorCPD:
  als_optimize,
  direct,
  random_CPD,
  random_CPD_square_network,
  row_norm,
  reconstruct,
  had_contract

## This function primes the internal indices of a 2d
## network. Doing so allows one to compute the norm
## of the ising network more easily
function replace_inner_w_prime_loop(tn)
  ntn = deepcopy(tn)
  for i in 1:(length(tn) - 1)
    cis = inds(tn[i])
    is = commoninds(tn[i], tn[i + 1])
    nis = [i ∈ is ? i' : i for i in cis]
    replaceinds!(ntn[i], cis, nis)
    cis = inds(tn[i + 1])
    nis = [i ∈ is ? i' : i for i in cis]
    replaceinds!(ntn[i + 1], cis, nis)
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

function compute_network_norm(tn)
  if length(tn) == 1
    return norm(tn.data_graph.vertex_data.values[1])
  end
  sising = tn.data_graph.vertex_data.values
  sisingp = replace_inner_w_prime_loop(sising)
  
  sqrs = sising[1] * sisingp[1]
  for i in 2:length(sising)
    sqrs = sqrs * sising[i] * sisingp[i]
  end
  return sqrt(sqrs[])
end

nx = 2
ny = 2
s = ITensorIndsNetwork(named_grid((nx, ny)); link_space=4);
tn = ITensorNetworks.random_tensornetwork(s)
s1 = subgraph(tn, ((2,1), (2,2), (1,2)))
s2 = subgraph(tn, ((1,1),))

r1 = Index(1, "CP_rank")
r2 = Index(1, "CP_rank")
fit = ITensorCPD.FitCheck(1e-3, 6, compute_network_norm(s1))
cp_s1 = ITensorCPD.als_optimize(ITensorCPD.random_CPD_ITensorNetwork(s1, r), r, fit);
tn1 = copy(tn)
tn1[1,1] = noprime(tn[1,1] * cp_s1[1] * (cp_s1[] * δ(r1, r1')))* prime(cp_s1[2]; tags="CP_rank")
tn[2,2] = tn[1,1] * cp_s1[1]