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

# function block_sample_factor_matrices(nsamps, probs, block_size, skip_fact)
#   blocked_factor_number = skip_fact == 1 ? 2 : 1
#   blocked_factor = probs[blocked_factor_number]
#   size_fast = length(blocked_factor)
#   nfactors = length(probs)
#   sampled_cols = Matrix{Int}(undef, (nsamps, nfactors - 1))

#   for i in 1:block_size+1:nsamps
#       sampled_cols[i,:] = sample_single_col_from_factors(skip_fact, probs)
#       m = 1
#       ## TODO add a check here to make sure that the blocking doesn't accidentally walk out of the range of the array
#       ## If it does, then I should push back the value to a place where it wont break things
#       ## One can accidnetally walk out of the array on the front side too (i.e. to negative)
#       ## And if block_size > size_fast theres also an issue.
#       start = sampled_cols[i, 1]
#       start = start + block_size > size_fast ? start - block_size : start
#       for j in i+1:i+block_size
#           if j > nsamps && break end
#           sampled_cols[j,:] = [start + m, sampled_cols[i,2:end]...]
#           m += 1
#       end
#   end
#   return sampled_cols
# end


function block_sample_factor_matrices(nsamps, probs, block_size, skip_fact)
  nfactors = length(probs)
  sampled_cols = Matrix{Int}(undef, (nsamps, nfactors - 1))

  ## Here I am going to fuse the blocked factor into a fixed number of 
  ## blocks (nearly) evenly divided and sum the probabilities to make 
  ## a block-wise probability array
  blocked_factor_number = skip_fact == 1 ? 2 : 1
  blocked_factor = probs[blocked_factor_number]
  size_fast = length(blocked_factor)

  nblocks = size_fast ÷  block_size
  resid = size_fast % block_size
  block_prob = eltype(probs)(undef, nblocks)
  block_sizes = Vector{Int}([1])
  m = 1
  for i in 1:nblocks
    block_prob[i] = sum(blocked_factor[m:(m + block_size + (i ≤ resid) - 1)])
    m += block_size + (i ≤ resid)
    push!(block_sizes, m)
  end

  ## Next I will grab a sample of the other factor matrices and fix them, then I will grab a block sample.
  ## I will then expand the block sample into all the elements in the block

  m = 1
  for i in 1:(nsamps ÷ block_size)
      other_cols = sample_single_col_from_factors(skip_fact, probs)
      block_sample = sample([x for x in 1:length(block_prob)], Weights(block_prob))
      start_block = block_sizes[block_sample]
      end_block = block_sizes[block_sample + 1]
      for j in 1:end_block - start_block
        if m > nsamps
          m += 1
          break 
        end
        sampled_cols[m, :] = [start_block + j - 1, other_cols[2:end]...]
        m += 1
      end
  end
  if m < nsamps + 1
    for i in m:nsamps
      sampled_cols[i, :] = sample_single_col_from_factors(skip_fact, probs)
    end
  end
  
  return sampled_cols
end