using ITensors: ITensor, Index
using TensorOperations
using ITensorNetworks: ITensorNetworks, ITensorNetwork

## This is a specialized tensor product which combines the hadamard product with other tensor
## operations. Here we assume A and B both have a matching mode `had` and this mode will 
## be preserved in tensor result. For each value of the common mode, there is a subnewtork 
## contraction problem which we resolve efficiently in place. In the future this is a point of
## parallelization.
## for example C(i,k,r) = α * A(i,j,r) * B(j,r,k) = ∀ rᵢ C_rᵢ(i,k) = α * A_rᵢ(i,j) B_rᵢ(j,k)
function had_contract(A::ITensor, B::ITensor, had::Index; α = true)
    @assert NDTensors.datatype(A) == NDTensors.datatype(B)
    dataT = NDTensors.datatype(A)
    if had ∉ commoninds(A, B)
        return α .* (A * B)
    end
    position_of_had_A = findfirst(x -> x == had, inds(A))
    position_of_had_B = findfirst(x -> x == had, inds(B))
    slices_A = eachslice(array(A); dims = position_of_had_A)
    slices_B = eachslice(array(B); dims = position_of_had_B)

    @assert length(slices_A) == length(slices_B)
    inds_c = noncommoninds(A, B)
    elt = promote_type(eltype(A), eltype(B))
    is = Tuple(vcat(inds_c..., had))
    C = similar(A, is)
    
    slices_C = eachslice(array(C); dims = ndims(C))
    a_inds = [ind(A, x) for x = 1:ndims(A) if x != position_of_had_A]
    b_inds = [ind(B, x) for x = 1:ndims(B) if x != position_of_had_B]
    ## TODO parallelize over this loop.
    for i = 1:length(slices_A)
        ITensors.contract!(
            itensor(slices_C[i], inds_c),
            itensor(slices_A[i], a_inds),
            itensor(slices_B[i], b_inds),
            elt(α),
        )
    end
    return C
end

## TODO this is broken when some items have a rank but others do not.
## This is a specialized tensor product which combines the hadamard product with other tensor
## operations like the definition before. In this we contract a collection of tensors 
## and we do not require all tensors share the common index `had`. Any tensor that does not
## share this common index will be effectively replicated for every subnetwork contraction.
## for example D(i,k,r) = A(i,j,r) * B(j,r,k) C(k,l) = ∀ rᵢ D_rᵢ(i,l) = (A_rᵢ(i,j) B_rᵢ(j,k)) C(k,l)
## The variable `sequence` can be provided to specify the contraction sequence of the subnetwork problems.
## if sequence == nothing then the optimal path will be computed.
function had_contract(tensors, had::Index; α = true, sequence = nothing)
    had_tensors = Vector{Int}() 
    no_had = Vector{Int}() 

    # Find all the tensors that don't contain the hadamard index
    for (ten, i) in zip(tensors,  1:length(tensors))
        if had ∉ inds(ten)
            push!(no_had, i)
            continue
        end
        push!(had_tensors, i)
    end

    # For all that do, find the mode remember its position in the each index list
    positions_of_had = Dict(y => (findfirst(x -> x == had, inds(y))) for y in tensors[had_tensors])
    # Slice each tensor along the hadamard dimension
    slices = [eachslice(array(x); dims = positions_of_had[x]) for x in tensors[had_tensors]]
    slices_inds = [inds(x)[1:end .!= positions_of_had[x]] for x in tensors[had_tensors]]

    slice_0 = Vector{ITensor}(vcat(
        [itensor(slices[x][1], slices_inds[x]) for x = 1:length(had_tensors)],
        tensors[no_had]))

    # Find the best way to contract the tensor graph if neessary
    sequence =
        isnothing(sequence) ? ITensors.optimal_contraction_sequence(slice_0) : sequence

    # Contract the full list of tensors but only the first hadamard slice
    cslice = α .* contract(slice_0; sequence)

    # ## Right now I have to fill C with zeros because I hate empty tensor
    # Use the first slice to determine the final indices and make the full resulting tensor
    C = similar(cslice, (inds(cslice)..., had))
    #ITensor(zeros(eltype(cslice), dim(had) * dim(cslice)), (had, inds(cslice)...))
    slices_c = eachslice(array(C); dims = ndims(C))
    slices_c[1] .= cslice

    ## TODO concat lists in a memory free way? all things are already stored in memory so if I could 
    ## make a list of pointers to the start of all these lists that would be cheap.
    for i = 2:dim(had)
        slice = Vector{ITensor}(
            vcat([itensor(slices[x][i], slices_inds[x]) for x = 1:length(had_tensors)],
            tensors[no_had]))

        slices_c[i] .= array(
            α .* contract(slice;
                sequence,
            ),
        )
    end

    return C
end

## This function computes the optimal hadamard_contraction sequence for a collection of tensors provided to 
## it as a list of itensors
function optimal_had_contraction_sequence(tensors::Vector{<:ITensor}, had::Index)
    had_tensors = Vector{Int}() 
    no_had = Vector{Int}() 

    # Find all the tensors that don't contain the hadamard index
    for (ten, i) in zip(tensors,  1:length(tensors))
        if had ∉ inds(ten)
            push!(no_had, i)
            continue
        end
        push!(had_tensors, i)
    end

    # For all that do, find the mode remember its position in the each index list
    positions_of_had = Dict(y => (findfirst(x -> x == had, inds(y))) for y in tensors[had_tensors])
    # Slice each tensor along the hadamard dimension
    slices = [eachslice(array(x); dims = positions_of_had[x]) for x in tensors[had_tensors]]
    slices_inds = [inds(x)[1:end .!= positions_of_had[x]] for x in tensors[had_tensors]]

    slice_0 = Vector{ITensor}(vcat(
        [itensor(slices[x][1], slices_inds[x]) for x = 1:length(had_tensors)],
        tensors[no_had]))

    return ITensors.optimal_contraction_sequence(slice_0)
end

## This is a specialized tensor product which combines the hadamard product with other tensor
## operations like the definition before. In this we contract a collection of tensors 
## and we do not require all tensors share the common index `had`. Any tensor that does not
## share this common index will be effectively replicated for every subnetwork contraction.
## for example D(i,k,r) = A(i,j,r) * B(j,r,k) C(k,l) = ∀ rᵢ D_rᵢ(i,l) = (A_rᵢ(i,j) B_rᵢ(j,k)) C(k,l)
## The variable `sequence` can be provided to specify the contraction sequence of the subnetwork problems.
## if sequence == nothing then the optimal path will be computed.
function had_contract(
    network::ITensorNetwork,
    had::Index;
    α = true,
    sequence = nothing,
    alg = nothing,
)
    alg = isnothing(alg) ? "optimal" : alg

    tensors = [network...]
    had_tensors = Vector{Int}([])
    no_had = Vector{Int}([])

    for (ten, i) in zip(tensors, 1:length(tensors))
        if had ∉ inds(ten)
            push!(no_had, i)
            continue
        end
        push!(had_tensors, i)
    end

    positions_of_had = Dict(y => (findfirst(x -> x == had, inds(y))) for y in tensors[had_tensors])
    slices = [eachslice(array(x); dims = positions_of_had[x]) for x in tensors[had_tensors]]
    slices_inds = [inds(x)[1:end .!= positions_of_had[x]] for x in tensors[had_tensors]]

    slice_0 = ITensorNetwork(vcat(
        [itensor(slices[x][1], slices_inds[x]) for x = 1:length(had_tensors)],
        tensors[no_had])
    )

    sequence =
        isnothing(sequence) ? ITensorNetworks.contraction_sequence(slice_0; alg) : sequence

    cslice = α .* contract(slice_0; sequence)

    C = similar(cslice, (inds(cslice)..., had))

    slices_c = eachslice(array(C); dims = ndims(C))
    slices_c[1] .= cslice

    for i = 2:dim(had)
        slice = ITensorNetwork(
            vcat([itensor(slices[x][i], slices_inds[x]) for x = 1:length(had_tensors)],
            tensors[no_had]))

        slices_c[i] .= array(
            α .* contract(slice;
                sequence,
            ),
        )
    end

    return C
end

## This function computes the optimal hadamard_contraction sequence for a collection of tensors provided to 
## it as an ITensorNetwork
function optimal_had_contraction_sequence(network::ITensorNetwork, had::Index; alg = nothing)
    alg = isnothing(alg) ? "optimal" : alg

    tensors = [network...]
    had_tensors = Vector{Int}([])
    no_had = Vector{Int}([])

    for (ten, i) in zip(tensors, 1:length(tensors))
        if had ∉ inds(ten)
            push!(no_had, i)
            continue
        end
        push!(had_tensors, i)
    end

    positions_of_had = Dict(y => (findfirst(x -> x == had, inds(y))) for y in tensors[had_tensors])
    slices = [eachslice(array(x); dims = positions_of_had[x]) for x in tensors[had_tensors]]
    slices_inds = [inds(x)[1:end .!= positions_of_had[x]] for x in tensors[had_tensors]]

    slice_0 = ITensorNetwork(vcat(
        [itensor(slices[x][1], slices_inds[x]) for x = 1:length(had_tensors)],
        tensors[no_had])
    )

    return ITensorNetworks.contraction_sequence(slice_0; alg)
end

## This function is special tensor product derived from the khatri-rao product
## each tensor must be a matrix with one matching mode. The hadamard product
## will be computed over the common mode. Canonically the outer product of the 
## other modes would be computed to compute the hadamard product.
## i.e. C(i,j,r) = A(i,r) ⊙ B(j,r). 
## This operation can be mapped to the contraction with a ij x ij identity matrix
## C(i,j,r) = I(i,j, i',j') A(i',r) B(j',r)
## This function computes a portion of this tensor product by providing a list 
## entries in this identity matrix via the tensor `pivots`
## we assume pivots gives the value of the combined `ij` index
function pivot_hadamard(A::ITensor, B::ITensor, had::Index, pivots::ITensor)
    @assert NDTensors.datatype(A) == NDTensors.datatype(B)
    @assert (had ∈ commoninds(A, B) )
    @assert ndims(A) == 2
    @assert ndims(B) == 2

    i = ind(A,1)
    j = ind(B,1)
    npivs = column_to_multi_coords(data(pivots), dim.((i,j)))

    return itensor((array(A)[npivs[:,1], :] .* array(B)[npivs[:,2], :]), inds(pivots)[end], had)
end

## This function is special tensor product derived from the khatri-rao product
## each tensor must be a matrix with one matching mode. See above for a full description.
## This function works for a list of tensors to be fused via the Khatri-Rao product.
function pivot_hadamard(tensors, had::Index, pivots::ITensor)
    for tensor in tensors
        @assert had == ind(tensor, 2)
    end

    ## Right now assume only one common ind.
    is = [commonind(pivots,x) for x in tensors]

    npivs = column_to_multi_coords(data(pivots), dim.(is))
    prod = ones(eltype(tensors[1]), size(npivs)[1], dim(had))
    for (tensor, i) in zip(tensors, 1:length(tensors))
        prod .*= array(tensor)[npivs[:,i], :]
    end
    
    return itensor(prod, inds(pivots)[end], had)
end

## This function is special tensor product derived from the khatri-rao product
## each tensor must be a matrix with one matching mode. See above for a full description.
## This function works for a list of tensors to be fused via the Khatri-Rao product.
## This will assume that the pivots are concatinated into an array
function pivot_hadamard(tensors::Vector{<:ITensor}, had::Index, pivots::Matrix, piv_ind::Union{<:Nothing, <:Index} = nothing)
    for tensor in tensors
        @assert had == ind(tensor, 2)
    end

    npivs = size(pivots)[1]
    prod = ones(eltype(tensors[1]), npivs, dim(had))
    for (tensor, i) in zip(tensors, 1:length(tensors))
        prod .*= array(tensor)[pivots[:,i], :]
    end
    
    piv_ind = isnothing(piv_ind) ? Index(npivs, "PivIdx") : piv_ind
    return itensor(prod, piv_ind, had)
end