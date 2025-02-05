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
function ising_network(
  eltype::Type, s::IndsNetwork, beta::Number; h::Number=0.0, szverts=nothing
)
  s = insert_linkinds(s; link_space=2)
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

##########################################

beta = 0.41
h = 0
s = IndsNetwork(named_grid((nx, ny)); link_space=2);

tn = ising_network(Float64, s, beta; h);
using NDTensors
r1 = Index(10, "CP_rank")
r2 = Index(10, "CP_rank")
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

A = s1[2, 2]
Ar = cpopt[1] * s1[2, 2]
ITensorCPD.had_contract(Ar, cpopt[2], r1)

r1 = Index(100, "CP_rank")
cp = ITensorCPD.random_CPD_ITensorNetwork(s1, r1);
@time ITensorCPD.als_optimize(cp, r1, fit);

cp = ITensorCPD.random_CPD_square_network(sising, r1);
@time cpopt = ITensorCPD.als_optimize(cp, r1, fit);

s3 = subgraph(tn, ((1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1)))
s4 = subgraph(tn, ((1, 2), (2, 2), (2, 1), (3, 1)))
sising = s1.data_graph.vertex_data.values
sisingp = replace_inner_w_prime_loop(sising)
sqrs = sising[1] * sisingp[1]
for i in 2:length(sising)
  sqrs = sqrs * sising[i] * sisingp[i]
end
sqrt(sqrs[])
r1 = Index(1000, "CP_rank")
fit = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
cp = ITensorCPD.random_CPD_ITensorNetwork(s1, r1);
@time ITensorCPD.als_optimize(cp, r1, fit);

nx = 3
ny = 3
nz = 3
box = named_grid((nx, ny, nz))

s = IndsNetwork(box; link_space=2)
tn_box = ising_network(Float64, s, beta)
s1 = subgraph(
  tn_box,
  ((1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 2, 3), (1, 3, 3), (1, 3, 2), (1, 3, 1), (1, 2, 1)),
);
s2 = subgraph(
  tn_box,
  ((2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 2, 3), (2, 3, 3), (2, 3, 2), (2, 3, 1), (2, 2, 1)),
);

s3 = subgraph(
  tn_box,
  (
    (2, 1, 1),
    (2, 1, 2),
    (2, 1, 3),
    (2, 2, 3),
    (2, 3, 3),
    (2, 3, 2),
    (2, 3, 1),
    (2, 2, 1),
    (1, 1, 1),
    (1, 1, 2),
    (1, 1, 3),
    (1, 2, 3),
    (1, 3, 3),
    (1, 3, 2),
    (1, 3, 1),
    (1, 2, 1),
  ),
);
r1 = Index(10, "CP_rank")
cp = ITensorCPD.random_CPD_ITensorNetwork(s1, r1);
sising = s1.data_graph.vertex_data.values
sisingp = replace_inner_w_prime_loop(sising)
sqrs = sising[1] * sisingp[1]
for i in 2:length(sising)
  sqrs = sqrs * sising[i] * sisingp[i]
end
sqrt(sqrs[])
fit = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
cpopt = ITensorCPD.als_optimize(cp, r1, fit);

sising = s2.data_graph.vertex_data.values
sisingp = replace_inner_w_prime_loop(sising)
sqrs = sising[1] * sisingp[1]
for i in 2:length(sising)
  sqrs = sqrs * sising[i] * sisingp[i]
end
fit = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]))
r2 = Index(50, "CP_rank_s2")
@time ITensorCPD.als_optimize(ITensorCPD.random_CPD_ITensorNetwork(s2, r2), r2, fit);

sising = s3.data_graph.vertex_data.values
sisingp = replace_inner_w_prime_loop(sising)
sqrs = sising[1] * sisingp[1]
for i in 2:length(sising)
  sqrs = sqrs * sising[i] * sisingp[i]
end
fit = ITensorCPD.FitCheck(1e-3, 6, sqrt(sqrs[]));
r2 = Index(40, "CP_rank_s2")
cpopt = ITensorCPD.als_optimize(ITensorCPD.random_CPD_ITensorNetwork(s3, r2), r2, fit);
norm(ITensorCPD.reconstruct(cpopt) - contract(s3)) / sqrt(sqrs[])
