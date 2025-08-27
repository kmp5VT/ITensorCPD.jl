using LinearAlgebra, StatsBase

function compute_leverage_score_probabilitiy(A, row::Index)
  ## This only works on matrices for now.
  @assert ndims(A) == 2
  q, _ = qr(A, row)
  ITensors.hadamard_product!(q, q, q)
  ni = dim(q, 1)
  return [sum(array(q)[i,:]) for i in 1:ni]
end

function samples_from_probability_vector(PW::Vector, samples)
  return Vector{Int}([sample([x for x in 1:length(PW)], Weights(PW)) for i in 1:samples])
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