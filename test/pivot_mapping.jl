using ITensors, ITensors.NDTensors
using ITensorCPD: column_to_multi_coords
# TEST 1
@testset "Check Pivot Map function" begin
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
    for p in eachindex(pivot_columns)
      pivot_idx = pivot_columns[p]
      for a_idx in 1:i
        bc_coord = bc_coordinates[p,:]
        @test  T_matrix[a_idx, pivot_idx] == T[a_idx, bc_coord...]
      end
    end

# TEST 2
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

    for p in eachindex(pivot_columns)
      pivot_idx = pivot_columns[p]
      for c_idx in 1:k
        abd_coord = abd_coordinates[p,:]
        @test T_matrix[c_idx, pivot_idx] == T[abd_coord[1], abd_coord[2], c_idx, abd_coord[3]]
      end
    end

    ### TODO add a test to compare had_contract
    i,j = Index.((100,150))
    m = Index(20, "had")

    A = random_itensor(i,m)
    B = random_itensor(j,m)
    exact_had = had_contract(A,B, m)

    p = [x for x in 1:100*150]
    l = Index(length(p), "Piv")
    P = itensor(NDTensors.tensor(Diag(p), (i,j,l,)))

    pivs = ITensorCPD.column_to_multi_coords(data(P), dim.((i,j)))
    sampled_had = Array{eltype(B)}(undef, (dim(l), dim(m)))
    for i in 1:dim(l)
      sampled_had[i,:] = array(exact_had)[pivs[i,1], pivs[i,2], :]
    end

    @test norm(vec(sampled_had) - exact_had.tensor.storage.data ) ≈ 0.0

    p = [rand(1:100*150) for x in 1:40]
    l = Index(length(p), "Piv")
    P = itensor(NDTensors.tensor(Diag(p), (i,j,l,)))

    pivs = ITensorCPD.column_to_multi_coords(data(P), dim.((i,j)))
    sampled_had = Array{eltype(B)}(undef, (dim(l), dim(m)))
    for i in 1:dim(l)
      sampled_had[i,:] = array(exact_had)[pivs[i,1], pivs[i,2], :]
    end

    @test norm(array(ITensorCPD.pivot_hadamard(A, B, m, P)) - sampled_had) ≈ 0.0

    @test norm(ITensorCPD.pivot_hadamard([A, B], m, P)  - ITensorCPD.pivot_hadamard(A, B, m, P)) ≈ 0.0

    k = Index(200)
    C = random_itensor(k,m)
    exact_had = had_contract([A,B,C], m)

    p = [rand(1:100*150*200) for x in 1:200]
    l = Index(length(p), "Piv")
    P = itensor(NDTensors.tensor(Diag(p), (i,j,k,l,)))

    pivs = ITensorCPD.column_to_multi_coords(data(P), dim.((i,j,k)))
    sampled_had = Array{eltype(B)}(undef, (dim(l), dim(m)))
    for i in 1:dim(l)
      sampled_had[i,:] = array(exact_had)[pivs[i,1], pivs[i,2], pivs[i,3], :]
    end
    norm(sampled_had - array(ITensorCPD.pivot_hadamard([A, B, C], m, P))) ≈ 0.0
end
