using StatsBase, SparseArrays
using LinearAlgebra: qr,norm
## generating a sparse embedding
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
function sparse_sign_matrix(d::Int, m::Int, zeta::Int)
    nnz=m*zeta
    vals = Array{Float64}(undef, nnz)
    rows = Array{Int32}(undef, nnz)          
    colstarts = Array{Int32}(undef, m+1)
    ccall((
        :sparse_sign,
        "/Users/israafakih/Downloads/Tensors/ITensorCPD.jl/src/algebra/libsparse_sign.so"
        ),
        Cvoid,
        (Cint, Cint, Cint, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
        d, m, zeta, vals, rows, colstarts
    )
    cols = repeat(0:m-1, inner=zeta)
    return sparse(rows .+ 1, cols .+ 1, vals, d, m)


end

##Randomized QR factorization
function SEQRCS(A,l,s,k,t)
    println("RandQR called!")
    n=size(A,2)
    omega = sparse_sign_matrix(l,n,s)
    B =  A*omega'
    F = qr(B, Val(true))  # pivoted QR
    _, _, p_B = F.Q, F.R, F.p
    p_B=p_B[1:t]
    println("The size of B is", size(B))
    println(k)
    rows_sel = omega[p_B,:]
    indices = findall(col -> any(col .!= 0), eachcol(rows_sel))
    A_tilde = A[:,indices]
    println("The size of A_tilde is", length(indices))
    F = qr(A_tilde, Val(true)) 
    Q, R, p_A = F.Q, F.R, F.p
    r_indices = setdiff(1:n,indices)
    p = vcat(indices[p_A],r_indices)
    A_tilde_2 = A[:,r_indices]
    R = hcat(R,Q'*A_tilde_2)
    return Q,R,p

end