using LinearAlgebra, StatsBase

function compute_leverage_score_probabilitiy(A, row::Index)
  ## This only works on matrices for now.
  @assert ndims(A) == 2
  q, _ = qr(A, row)
  ITensors.hadamard_product!(q, q, q)
  ni = dim(q, 1)
  return [sum(array(q)[i,:])  for i in 1:ni] ./ minimum(dims(A))
end

function samples_from_probability_vector(PW::Vector, samples)
  return Vector{Int}([sample([x for x in 1:length(PW)], Weights(PW)) for i in 1:samples])
end

function sample_single_col_from_factors(skip_factor, probs)
  return [sample([x for x in 1:length(prob)], Weights(prob)) for prob in probs[1:end .!= skip_factor]]
end
## uniform sampling of each factor matrix (no blocking)
## nsamps = Number of samples 
## skip_factor = which factor will be ignored
## prob = list of probabilities for all factor matrices
function sample_factor_matrices(nsamps, skip_factor, probs)
    nfactors = length(probs)
    sampled_cols = Matrix{Int}(undef, (nsamps, nfactors - 1))
    m = 1
    for i in 1:nfactors-1
        m = i == skip_factor ? m + 1 : m
        sampled_cols[:,i] = ITensorCPD.samples_from_probability_vector(probs[m], nsamps)
        m += 1
    end
    return sampled_cols
end

function block_sample_factor_matrices(nsamps, probs, block_size, skip_fact, size_fast)
  nfactors = length(probs)
  sampled_cols = Matrix{Int}(undef, (nsamps, nfactors - 1))

  for i in 1:block_size+1:nsamps
      sampled_cols[i,:] = sample_single_col_from_factors(skip_fact, probs)
      m = 1
      ## TODO add a check here to make sure that the blocking doesn't accidentally walk out of the range of the array
      ## If it does, then I should push back the value to a place where it wont break things
      ## One can accidnetally walk out of the array on the front side too (i.e. to negative)
      ## And if block_size > size_fast theres also an issue.
      start = sampled_cols[i, 1]
      start = start + block_size > size_fast ? start - block_size : start
      for j in i+1:i+block_size
          if j > nsamps && break end
          sampled_cols[j,:] = [start + m, sampled_cols[i,2:end]...]
          m += 1
      end
  end
  return sampled_cols
end