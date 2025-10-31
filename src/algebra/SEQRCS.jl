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

using TimerOutputs

"""
SEQRCS(A,l,s,k,t)

Performs Randomized QR to a matrix 'A' for a given rank 'k'
# Arguments
'A': target tensor
'mode': mode along which to matricize
'i': index of the mode
'l': embedding dimension
's': sparsity parameter
'k': rank of QR on 'A'
't': rank of QR on sketched matrix 'A_sk'


"""
function SEQRCS(A:: ITensor,mode::Int,i,l,s,t)
    timer = TimerOutput()

    Ris = uniqueinds(A, i)         
    n = dim(Ris)

    # Generate sparse embedding
    @timeit timer "Omega" omega = sparse_sign_matrix(l,n,s)

    # Sketch the matrix and applying QR 
    @timeit timer "sketching" A_sk = sketched_matricization(A, mode , omega')
    
    @timeit timer "QR1" begin 
    _, _, p_sk = qr(A_sk, ColumnNorm())  
    p_sk=p_sk[1:t]
    println("The size of A_sk is $(size(A_sk))")
    end

    ## Map back  pivots from 'A_sk' to 'A' and forming 'A_subset'
    @timeit timer "row select" begin
    rows_sel = omega[p_sk,:]
    @timeit timer "findall" indices = findall(col -> any(!=(0), col), eachcol(rows_sel))
    indices_ind = Index(length(indices),"ind")
    indices_tensor = itensor(Int, indices, indices_ind)
    @timeit timer "fused_flatten" A_subset = fused_flatten_sample(A, mode, indices_tensor)
    A_subset = array(A_subset)
    println("The size of A_subset is $(length(indices))")
    end

    ## Perform QR on A_subset to get final 'k' pivots
    @timeit timer "qr2" begin
    Q, R, p_subset = qr(A_subset, ColumnNorm()) 
    rem_indices = setdiff(1:n,indices)
    p = vcat(indices[p_subset],rem_indices)
    end

    ## Form  A_rem to get the factor 'R' 
    ## We can remove this part no need to get Q and R
    ## but keeping it just to make sure that the function is performing well
    @timeit timer "Reconstruct" begin
    rem_indices_ind = Index(length(rem_indices),"rem_ind")
    rem_indices_tensor = itensor(rem_indices, rem_indices_ind)
    A_rem = fused_flatten_sample(A, mode, rem_indices_tensor)
    A_rem = matrix(A_rem)
    R = hcat(R,Q'*A_rem)
    end
    # @show timer
    return Q,R,p

end