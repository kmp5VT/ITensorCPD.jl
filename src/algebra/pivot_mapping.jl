using ITensors
using ITensors: NDTensors

# The function mapping the column index to (b, c) (only works for order-3 tensor)
function column_to_bc_coords(col_indices, b)
    coords = []
    for col_idx in col_indices
        linear_idx = col_idx - 1
        b_idx = (linear_idx % b) + 1  
        c_idx = (linear_idx ÷ b) + 1   
        push!(coords, (b_idx, c_idx))
    end
    return coords
end

# The function mapping the column index to multiple coordinates 
function column_to_multi_coords(col_indices, dims)
    # col_indices: column pivot indices of matrix reshaped from tensor
    # dims: dimension for residual modes (b,c,d...)
    num_samples = length(col_indices)
    coords = similar(col_indices, (num_samples, length(dims)))

    for (col_idx, i) in zip(col_indices, 1:num_samples)
        linear_idx = col_idx - 1
        coord = Vector{Int}()
        remaining = linear_idx
        
        for dim in dims
            push!(coord, (remaining % dim) + 1)
            remaining = remaining ÷ dim
        end
        coords[i,:] = coord
    end
    return coords
end

## This function maps the multi-index data from the factor matricizes
## back to the columns of the matricized tensor
## sizes is a tuple of indices for the modes of the factor matrices.
## pivots is a matrix of where the first mode is the number of samples and the 
## second mode is the number of matrices being sampled.
function multi_coords_to_column(sizes::Union{<:Tuple,<:Vector}, pivots::Matrix)
  strides = [prod(sizes[1:end-s]) for s in 1:length(sizes)]
  npivots = size(pivots)[1]
  indices = [pivots[v,:] for v in 1:npivots]
  return [sum((i[end:-1:1] .- 1) .* strides) + 1 for i in indices]
end

## α is the sampled index from the matricized tensor
## sizes are the dimensions of the tensor and 
## k is the mode being flattened.
function transform_alpha_to_vectorized_tensor_position(α, extent, stride)::Int
  T = floor((α-1) / stride)
  return T * stride * extent + (α - T * stride)
end

## This function will take a higher-order tensor and a list of pivots
## and matricizes it along the `k`the dimension returning a matrix of size dim(k) x num_samples
function fused_flatten_sample(T::ITensor, k::Int, pivots::ITensor)
  v = vec(NDTensors.data(T))
  idx = ind(T, k)
  As = similar(T, (idx, inds(pivots)[end]))
  stride = strides(T.tensor)[k]
  pos = map(x -> ITensorCPD.transform_alpha_to_vectorized_tensor_position(x, dim(idx), stride), NDTensors.data(pivots))
  ## Not sure which is faster
  map!(i -> (@view v[pos .+ stride * (i - 1)]), eachrow(array(As)), 1:dim(idx))
  # As_slice = eachrow(array(As))
  # for i in 1:dim(idx)
  #   As_slice[i] .= @view v[pos .+ stride * (i - 1)]
  # end
  return As
end

## This function takes a higher order tensor and a sparse matrix
## and gives the sketched matricization of the tensor

function  sketched_matricization(T::ITensor, k::Int, omega)
  v = vec(NDTensors.data(T))
  l = size(omega,2)
  idx = ind(T, k)
  As = similar(NDTensors.similartype(NDTensors.data(T), (1,2)), dim(idx), l)
  As_slice = eachcol(As)
  Om_slice = eachcol(omega)
  stride = strides(T.tensor)[k]
  for j in 1:l
    pos = map(x -> ITensorCPD.transform_alpha_to_vectorized_tensor_position(x, dim(idx), stride), findall(!iszero,@view Om_slice[j]))
    As_slice[j] .= [sum(@view v[pos .+ stride* (i-1)]) for i in 1:dim(idx)]
  end
  return As
end