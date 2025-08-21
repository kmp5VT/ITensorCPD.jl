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
    coords = []
    for col_idx in col_indices
        linear_idx = col_idx - 1
        coord = []
        remaining = linear_idx
        
        for dim in dims
            push!(coord, (remaining % dim) + 1)
            remaining = remaining ÷ dim
        end
        push!(coords, tuple(coord...))
    end
    return coords
end
