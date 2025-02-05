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