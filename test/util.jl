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
