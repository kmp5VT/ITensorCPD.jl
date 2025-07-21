using ITensors
using NDTensors

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

function pivot_hadamard(B::ITensor, C::ITensor, r::Index, pivot_indices::Array)
    # We first assume that the first mode of given tensors should match with the given index r
    if inds(B)[1] != r || inds(C)[1] != r
        throw("The first dimension of input iTensors does not match with the given r.")
    end 

    # Map matrix pivot indices to tensor indices
    pivot_dims = (dim(inds(B)[2]), dim(inds(C)[2]));
    bc_coordinates = column_to_multi_coords(pivot_indices, pivot_dims);
    
    # Hadamard product of B and C in selected pivot indices
    l = Index(length(pivot_indices), "l")  
    T = ITensor(r, l)
    for i in 1:dim(r)
        for j in 1:dim(l)
            T[i,j] = B[i, bc_coordinates[j][1]] * C[i, bc_coordinates[j][2]]
        end
    end

    return T  
end


k_dim, j_dim, r_dim = 6, 5, 4;
k = Index(k_dim, "k");
j = Index(j_dim, "j");
r = Index(r_dim, "r");
B = randomITensor(r,j);
C = randomITensor(r,k);

pivots = [27, 4, 13];
T = pivot_hadamard(B, C, r, pivots);


# TEST 1
let 
    # Random tensor
    i, j, k, l = 4, 5, 6, 3;
    a = Index(i, "a");
    b = Index(j, "b");
    c = Index(k, "c");
    d = Index(l, "d");
    T = randomITensor(a, b, c, d);

    # Reshape tensor to (a, bc) (column major in Julia I think)
    T_ndt = tensor(T);
    T_matrix = reshape(T_ndt, (dim(a), dim(b) * dim(c) * dim(d)));

    pivot_columns = [1, 4, 7, 12, 29, 30, 8, 21, 17, 42, 62, 86, 72];
    col_dims = (j,k,l);
    bc_coordinates = column_to_multi_coords(pivot_columns, col_dims);

    # Examination of index mapping
    flag = 0
    for p in eachindex(pivot_columns)
        pivot_idx = pivot_columns[p]
        for a_idx in 1:i
            bc_coord = bc_coordinates[p]
            if T_matrix[a_idx, pivot_idx] != T[a_idx, bc_coord...]
                flag = 1
                println("Error of mapping!")
            end
        end
    end

    if flag == 0
        println("Test succeeds!")
    end
end

# TEST 2
let
    # Random tensor
    i, j, k, l = 3, 6, 8, 2;
    a = Index(i, "a");
    b = Index(j, "b");
    c = Index(k, "c");
    d = Index(l, "d");
    T = randomITensor(a, b, c, d);

    # Reshape tensor to (c,abd) (column major in Julia I think)
    ris = [a,b,d];
    T_matrix = reshape(array(T, (c, ris...)), (dim(c), dim(ris)));

    pivot_columns = [1, 3, 7, 14, 29, 30, 10, 22, 35, 8, 11];
    col_dims = (i,j,l);
    abd_coordinates = column_to_multi_coords(pivot_columns, col_dims);

    flag = 0
    for p in eachindex(pivot_columns)
        pivot_idx = pivot_columns[p]
        for c_idx in 1:k
            abd_coord = abd_coordinates[p]
            if T_matrix[c_idx, pivot_idx] != T[abd_coord[1], abd_coord[2], c_idx, abd_coord[3]]
                flag = 1
                println("Error of mapping!")
            end
        end
    end

    if flag == 0
        println("Test succeeds!")
    end
end
