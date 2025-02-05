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

using ITensorNetworks: vertices
nx,ny,nz = 3,3,3
g = named_grid((nx,ny,nz));
s = siteinds("S=1/2", g);
χ = 2
ψ = ITensorNetworks.random_tensornetwork(s; link_space = χ);
ψψ = ITensorNetwork([v => ψ[v]*noprime(prime(dag(ψ[v])); tags = "Site") for v in vertices(ψ)]);
#ψψ= ITensorNetworks.combine_linkinds(ψψ)
op_str = "Z"

function layer(nz, tn)
  return subgraph(
  tn,
  ((1,1,nz), (1, 2, nz), (1, 3, nz), (2, 3, nz), (3, 3, nz), (3,2,nz), (3, 1, nz), (2, 1, nz)),
    )
end


function cp_of_layers(nz, ψψ, r; cores=nothing)
  if isnothing(cores)
    cores = [ψψ[2,2,nz],]
  end

  
  lay = layer(nz, ψψ);
  cp = ITensorCPD.random_CPD_ITensorNetwork(lay, r);
  fit = ITensorCPD.FitCheck(1e-3, 6, compute_network_norm(lay))
  opt = ITensorCPD.als_optimize(cp, r, fit);

  return opt
end

function t()
  ps = Vector{ITensor}()
  for core_tensor in cores
    tensors_to_core = Int64[]
    for y in inds(core_tensor)
      connection = findfirst(x -> y ∈ inds(x), opt.factors)
      !isnothing(connection) && push!(tensors_to_core, connection)
    end

    @show tensors_to_core
    p = copy(core_tensor)
    for x in tensors_to_core
      p = ITensorCPD.had_contract(p, opt[x], r)
    end
    push!(ps, p)
    # push!(ps, noprime((delta(r1')* cpopt[]) * p; tags=tags(r1)))
  end
end

function fuse_ring_and_core(layer_1, layer_2, r1::Index, r2::Index)
  core = layer_1[2] * layer_2[2]
  had = fill!(ITensor(eltype(core), r1, r2), one(eltype(core)))
  @show typeof(layer_2[1])
  for x in layer_1[1]
    is = ind(x, 2)
    next_layer_friend = findfirst(y -> ind(y, 2) == is,  layer_2[1])
    if !isnothing(next_layer_friend)
      had = hadamard_product(had, x * layer_2[1][next_layer_friend])
    end
  end
  positions_of_r1 = findfirst(x -> x == r1, inds(had))
  slices = eachslice(array(had); dims=positions_of_r1)
  λ = layer_1[3]
  for x in 1:length(slices)
    slices[x] .*= λ[x]
  end
  return core, had
end

v_centre = (2,2,2)
ITensors.disable_warn_order()
num = ψ[v_centre] * dag(prime(ψ[v_centre])) * ITensors.op(op_str, s[v_centre])
denom = ψ[v_centre] * dag(prime(ψ[v_centre])) *  ITensors.op("I", s[v_centre])

# for y in inds(core_tensor)
#   connection = findfirst(x -> y ∈ inds(x), cp_layer_1.factors)
#   !isnothing(connection) && push!(tensors_to_core, connection)
# end
# p = copy(core_tensor)
# core_vecs = [cp_layer_1[x] for x in tensors_to_core];
# for x in tensors_to_core
#   p = ITensorCPD.had_contract(p, cp_layer_1[x], r1)
# end
# cp_ring = ITensorCPD.had_contract(setdiff(cp_layer_1.factors, core_vecs), r1)
# cp_flat_layer_1 = noprime((delta(r1') * cp_layer_1.λ) * p; tags=tags(r1)) * cp_ring
r1 = Index(500, "CP_rank_l1")
cp_layer_1 = cp_of_layers(1, ψψ, r1);
cp_flat_layer_1 = contract([ITensorCPD.reconstruct(cp_layer_1), ψψ[2,2,1]])
# flat_layer_1 = contract(layer(1, ψψ)..., ψψ[2,2,1])
# norm(cp_flat_layer_1 - flat_layer_1) / norm(flat_layer_1)

r3 = Index(500, "CP_rank_l3")
cp_layer_3 = cp_of_layers(3, ψψ, r3);
cp_flat_layer_3 = contract([ITensorCPD.reconstruct(cp_layer_3), ψψ[2,2,3]])
# flat_layer_3 = contract(layer(3, ψψ)..., ψψ[2,2,3])
# norm(cp_flat_layer_3 - flat_layer_3) / norm(flat_layer_3)

r2 = Index(500, "CP_rank_l2")
cp_layer_2 = cp_of_layers(2, ψψ, r2);
l = length(cp_layer_2.factors)
# half = ITensorCPD.had_contract(cp_layer_2[1:l ÷ 2], r2)
# for r in 1:dim(r2)
#   array(cp_layer_2[l ÷ 2 + 1])[r,:] .*= cp_layer_2[][r]
# end
# half2 = ITensorCPD.had_contract(cp_layer_2[l ÷ 2 + 1:l], r2)
#flat_layer_2 = contract(layer(2, ψ)..., ψ[2,2,2])
for r in 1:dim(r2)
  array(cp_layer_2[1])[r,:] .*= cp_layer_2[][r]
end
fill!(cp_layer_2[], 1.0)

# cp_flat_layer_2 = ITensorCPD.reconstruct(cp_layer_2) * ψ[2,2,2]

cp_flat_layer_2_1 = cp_flat_layer_1 * ITensorCPD.had_contract(cp_layer_2[1:l ÷ 3], r2) 
cp_flat_layer_2_1 = cp_flat_layer_2_1 * ITensorCPD.had_contract(cp_layer_2[l ÷ 3+1: l ÷ 3 + l ÷ 3], r2) 
cp_flat_layer_2_2 = ITensorCPD.had_contract(cp_layer_2[l ÷ 3 + l ÷ 3 + 1 : l], r2) 
# cp_flat_layer_2_1 = cp_flat_layer_1 * cp_flat_layer_2_1
cp_flat_layer_2_2 = cp_flat_layer_2_2 * num
cp_flat_layer_1 = cp_flat_layer_2_1 * cp_flat_layer_2_2
cp_flat_layer_1 = cp_flat_layer_1 * cp_flat_layer_3
n_num = cp_flat_layer_1[]

cp_flat_layer_1 = contract([ITensorCPD.reconstruct(cp_layer_1), ψψ[2,2,1]])
cp_flat_layer_2_1 = ITensorCPD.had_contract(cp_layer_2[1:l ÷ 2], r2) 
cp_flat_layer_2_2 = ITensorCPD.had_contract(cp_layer_2[l ÷ 2 + 1 : l], r2) 
cp_flat_layer_2_1 = cp_flat_layer_1 * cp_flat_layer_2_1
cp_flat_layer_2_2 = cp_flat_layer_2_2 * denom
cp_flat_layer_1 = cp_flat_layer_2_1 * cp_flat_layer_2_2
cp_flat_layer_1 = cp_flat_layer_1 * cp_flat_layer_3
n_denom = cp_flat_layer_1[]

sz_cp = n_num / n_denom
# s = subgraph(ψψ, ((1,1,2), (1,2,2), (1,3,2), (2,1,2), (2,2,2), (2,3,2), (3,1,2), (3,2,2), (3,3,2))) 
# n= 0.0
# v = vertices(s)
# ss = [s[first(v)] * prime(s[first(v)], uniqueinds(s, first(v))),]
# for vert in v
#   @show vert
#   if vert == first(v)
#     continue
#   end
#   push!(ss ,[vert] * prime(s[vert], uniqueinds(s, vert)))
# end

# norm(cp_flat_layer_2 - flat_layer_2) / norm(flat_layer_2)

# ((flat_layer_1 * flat_layer_2 * flat_layer_3) * (flat_layer_1 * flat_layer_2 * flat_layer_3))[]
# ((cp_flat_layer_1 * cp_flat_layer_2_1 * cp_flat_layer_2_2 * cp_flat_layer_3) * (cp_flat_layer_1 * cp_flat_layer_2_1 * cp_flat_layer_2_2 * cp_flat_layer_3))[]

# results = Float64[]
# for cp_layer_2 in [cp_layer_2_num, cp_layer_2_denom]
#   core, had = fuse_ring_and_core(cp_layer_1, cp_layer_2, r1, r2)

#   core = ITensorCPD.had_contract(core, had, r2)

#   cp_layer_22 = (cp_layer_2[1], core, cp_layer_2[3])
#   core, had = fuse_ring_and_core(cp_layer_22, cp_layer_3, r2, r3)

#   core  = ITensorCPD.had_contract(core, had, r3)
#   result = (core * cp_layer_3[3])[]
#   push!(results, result)
# end
# @show results[1]/results[2]
# end

# core = core * cp_layer_3[2]
# had = fill!(ITensor(eltype(core), r2, r3), one(eltype(core)))
# for x in cp_layer_2[1]
#   is = ind(x, 2)
#   next_layer_friend = findfirst(y -> ind(y, 2) == is,  cp_layer_3[1])
#   if !isnothing(next_layer_friend) 
#     had = hadamard_product(had, x * cp_layer_3[1][next_layer_friend])
#   end
# end

# (core * had)[]

#################################
using ITensorNetworks
using ITensorNetworks: environment, contraction_sequence
using OMEinsumContractionOrders: OMEinsumContractionOrders
env_bp = environment(ψψ, [v_centre]; alg = "bp")
contraction_sequence_kwargs = (; alg = "sa_bipartite")
env_exact = environment(ψψ, [v_centre]; contraction_sequence_kwargs, alg = "exact")

bp_numer = contract([env_bp; ψ[v_centre]; dag(prime(ψ[v_centre])); ITensors.op(op_str, s[v_centre])]; sequence = "automatic")[]
bp_denom = contract([env_bp; ψ[v_centre]; dag(prime(ψ[v_centre])); ITensors.op("I", s[v_centre])]; sequence = "automatic")[]
exact_numer = contract([env_exact; ψ[v_centre]; dag(prime(ψ[v_centre])); ITensors.op(op_str, s[v_centre])]; sequence = "automatic")[]
exact_denom = contract([env_exact; ψ[v_centre]; dag(prime(ψ[v_centre])); ITensors.op("I", s[v_centre])]; sequence = "automatic")[]
sz_bp = bp_numer / bp_denom
sz_exact = exact_numer / exact_denom
sz_cp
@show sz_bp, sz_exact