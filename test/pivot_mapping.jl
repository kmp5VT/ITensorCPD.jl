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
    T = random_itensor(a, b, c, d);

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
    T = random_itensor(a, b, c, d);

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

    pivs = ITensorCPD.column_to_multi_coords(p, dims((i,j)))
    sampled_had = Array{eltype(B)}(undef, (dim(l), dim(m)))
    for i in 1:dim(l)
      sampled_had[i,:] = array(exact_had)[pivs[i,1], pivs[i,2], :]
    end

    @test norm(vec(sampled_had) - exact_had.tensor.storage.data ) ≈ 0.0

    p = [rand(1:100*150) for x in 1:40]
    l = Index(length(p), "Piv")
    pivs = column_to_multi_coords(p, dims((i,j)))
    
    sampled_had = Array{eltype(B)}(undef, (dim(l), dim(m)))
    for i in 1:dim(l)
      sampled_had[i,:] = array(exact_had)[pivs[i,1], pivs[i,2], :]
    end

    P = itensor(Int, pivs, l, Index(2))
    @test norm(array(ITensorCPD.pivot_hadamard(A, B, m, P)) - sampled_had) ≈ 0.0

    @test norm(ITensorCPD.pivot_hadamard([A, B], m, P)  - ITensorCPD.pivot_hadamard(A, B, m, P)) ≈ 0.0

    k = Index(200)
    C = random_itensor(k,m)
    exact_had = had_contract([A,B,C], m)

    p = [rand(1:100*150*200) for x in 1:200]
    l = Index(length(p), "Piv")
    pivs = ITensorCPD.column_to_multi_coords(p, dims((i,j,k)))

    P = itensor(Int, pivs, l, Index(3))

    sampled_had = Array{eltype(B)}(undef, (dim(l), dim(m)))
    for i in 1:dim(l)
      sampled_had[i,:] = array(exact_had)[pivs[i,1], pivs[i,2], pivs[i,3], :]
    end
    norm(sampled_had - array(ITensorCPD.pivot_hadamard([A, B, C], m, P))) ≈ 0.0
    @test all(array(ITensorCPD.fused_flatten_sample(exact_had, 4, P)) - sampled_had' .≈ 0)

    p = [rand(1:100*150*200) for x in 1:200]
    multi = column_to_multi_coords(p, dims((i,j,k)))
    @test all(p - ITensorCPD.multi_coords_to_column(dims((i,j,k)), multi) .== 0)

    ## Considering sparse matrix sketching
    cpd = ITensorCPD.random_CPD(T, 10)
    n = dim(inds(T)[1:end-1])
    m = dim(T,4)
    l=Int(round(3 * m * log(m)))
    s=Int(round(log(m)))
    vals = Array{Float64}(undef, n * s)
    rows = Array{Int32}(undef, n * s)
    omega = ITensorCPD.sparse_sign_matrix(l,n,s, rows, vals; omega=true,injective=false)
    @test norm(reshape(array(T), n,m)' * omega' - ITensorCPD.sketched_matricization(T, 4, omega)) < 1e-12
    @test norm(reshape(array(T), n,m)' * omega' - ITensorCPD.sketched_matricization(T, 4, l, rows, vals, s)) < 1e-12

    cprank = ITensorCPD.cp_rank(cpd)
    oh = ITensorCPD.omega_hadamard(cpd.factors[1:3], cprank, omega)
    exact_had = ITensorCPD.had_contract(cpd.factors[1:3], cprank)
    @test norm(array(oh)' - reshape(array(exact_had), (3 * 6 * 8, 10))' * omega') < 1e-12

    
    n = dim(inds(T)[1:end .!=2])
    m = dim(T,2)
    l=Int(round(3 * m * log(m)))
    s=Int(round(log(m)))
    vals = Array{Float64}(undef, n * s)
    rows = Array{Int32}(undef, n * s)
    omega = ITensorCPD.sparse_sign_matrix(l,n,s, rows, vals; omega=true,injective=false)

    oh = ITensorCPD.omega_hadamard(cpd.factors[1:end .!=2], cprank, omega)
    exact_had = ITensorCPD.had_contract(cpd.factors[1:end .!= 2], cprank)
    
    @test norm(array(oh)' - reshape(array(exact_had), (3 * 8 * 2, 10))' * omega') < 1e-12

    A = random_itensor(Index(rand(1:200)), cprank)
    B = random_itensor(Index(rand(2:200)), cprank)

    ia = dim(A, 1)
    ib = dim(B, 1)

    n = ia * ib
    m = dim(cprank)
    l=Int(round(3 * m * log(m)))
    s=Int(round(log(m)))
    vals = Array{Float64}(undef, n * s)
    rows = Array{Int32}(undef, n * s)
    omega = ITensorCPD.sparse_sign_matrix(l,n,s, rows, vals; omega=true,injective=false);

    krp = ITensorCPD.had_contract(A,B,cprank)
    krpm = reshape(array(krp), (ia*ib, m))
    omega * krpm
    oh = ITensorCPD.omega_hadamard([A,B], cprank, omega)
    @test all(array(oh) - omega * krpm .< 1e-10)
end
