using StatsBase
using SparseArrays: sparse
using LinearAlgebra

"""
Sparse_Emd(n,l,s) 

Generates oblivious sparrse embedding of size 'l' by 'n' with 's' non zero entries in each column
#Arguments
'n': number of columns
'l': number of rows (embedding dimension)
's': number of nonzeros in each column (sparsity parameter)

"""
function Sparse_Emd(n,l,s)
    omega = zeros(l,n)
    for i in 1:n
    indices = sample(1:l, s; replace=false)
    vals = rand([-1, 1], s) ./ sqrt(s)
    omega[indices,i]=vals
    end
    return omega
end

##Wrapping the sparse_sign file into a julia function
function sparse_sign_matrix(l::Int, n::Int, s::Int)
    nnz=n*s
    vals = Array{Float64}(undef, nnz)
    rows = Array{Int32}(undef, nnz)          
    colstarts = Array{Int32}(undef, n+1)
    ccall((
        :sparse_sign,
        libsparse
        ),
        Cvoid,
        (Cint, Cint, Cint, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
        l, n, s, vals, rows, colstarts
    )
    cols = repeat(0:n-1, inner=s)
    return sparse(rows .+ 1, cols .+ 1, vals, l, n)
end

"""
SEQRCS(A,l,s,k,t)

Performs Randomized QR to a matrix 'A' for a given rank 'k'

# Arguments
'A': input matrix
'l': embedding dimension
's': sparsity parameter
'k': rank of QR on 'A'
't': rank of QR on sketched matrix 'A_sk'


"""
function SEQRCS(A,l,s,k,t)
    n=size(A,2)

    # Generate sparse embedding
    omega = sparse_sign_matrix(l,n,s)

    # Sketch the matrix and applying QR 
    A_sk =  A*omega'
    F = qr(A_sk, ColumnNorm())  
    _, _, p_sk = F.Q, F.R, F.p
    p_sk=p_sk[1:t]
    println("The size of A_sk is", size(A_sk))

    ## Map back  pivots from 'A_sk' to 'A' and forming 'A_subset'
    rows_sel = omega[p_sk,:]
    indices = findall(col -> any(col .!= 0), eachcol(rows_sel))
    A_subset = A[:,indices]
    println("The size of A_subset is", length(indices))

    ## Perform QR on A_subset to get final 'k' pivots
    F = qr(A_subset, ColumnNorm()) 
    Q, R, p_subset = F.Q, F.R, F.p
    rem_indices = setdiff(1:n,indices)
    p = vcat(indices[p_subset],rem_indices)

    ## Form  A_rem to get the factor 'R'
    A_rem = A[:,rem_indices]
    R = hcat(R,Q'*A_rem)
    return Q,R,p

end