# function one_in_three_sat_network(
#   eltype::Type, s::IndsNetwork, link_space
# )
#   s = insert_linkinds(s; link_space)
#   tn = delta_network(eltype, s)
  
#   for edge in edges(tn)
#     i = inds(tn[edge])
#     deg_v1 = degree(tn, edge)
#     f11 = exp(beta * (1 + h / deg_v1 + h / deg_v2))
#     f12 = exp(beta * (-1 + h / deg_v1 - h / deg_v2))
#     f21 = exp(beta * (-1 - h / deg_v1 + h / deg_v2))
#     f22 = exp(beta * (1 - h / deg_v1 - h / deg_v2))
#     q = eltype[f11 f12; f21 f22]
#     w, V = eigen(q)
#     w = map(sqrt, w)
#     sqrt_q = V * ITensors.Diagonal(w) * inv(V)
#     t = itensor(sqrt_q, i, i')
#     tn[v1] = tn[v1] * t
#     tn[v1] = noprime!(tn[v1])
#     t = itensor(sqrt_q, i', i)
#     tn[v2] = tn[v2] * t
#     tn[v2] = noprime!(tn[v2])
#   end
#   return tn
# end

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

function norm_of_loop(s1::ITensorNetwork)
  sising = s1.data_graph.vertex_data.values
  sisingp = replace_inner_w_prime_loop(sising)

  sqrs = sising[1] * sisingp[1]
  for i in 2:length(sising)
    sqrs = sqrs * sising[i] * sisingp[i]
  end
  return sqrt(sqrs[])
end

function ring_inds(start::Int, nx::Int, ny::Int)
  inds = Vector{Tuple{Int,Int}}()
  for y in start:(ny-start+1)
      push!(inds, (start,y))
  end
  for x in start+1:(nx-start+1)
      push!(inds, (x, ny-start+1))
  end
  for y in (ny-start +1):-1:start
      push!(inds, (nx-start+1, y))
  end
  for x in (nx-start+1):-1:start+1
  push!(inds, (x, start))
  end
  return Tuple(unique!(inds))
end