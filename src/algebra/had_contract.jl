using ITensors: ITensor, Index
using TensorOperations
using ITensorNetworks: ITensorNetworks, ITensorNetwork

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
function had_contract(tensors::Vector{<:ITensor}, had::Index; α = true, sequence = nothing)
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
