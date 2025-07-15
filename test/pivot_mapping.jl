using ITensors
using NDTensors

# The function mapping the column index to (b, c)
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

# Random tensor
i, j, k = 4, 5, 6;
a = Index(i, "a");
b = Index(j, "b");
c = Index(k, "c");
T = randomITensor(a, b, c);

# Reshape tensor to (a, bc) (column major in Julia I think)
T_ndt = tensor(T);
T_matrix = reshape(T_ndt, (dim(a), dim(b) * dim(c)));

pivot_columns = [1, 4, 7, 12, 29, 30, 8, 21, 17];
bc_coordinates = column_to_bc_coords(pivot_columns, j, k);

# Examination of index mapping
for p in eachindex(pivot_columns)
    pivot_idx = pivot_columns[p]
    for a_idx in 1:i
        bc_coord = bc_coordinates[p]
        if T_matrix[a_idx, pivot_idx] != T[a_idx, bc_coord...]
            println("Error of mapping!")
        end
    end
end
