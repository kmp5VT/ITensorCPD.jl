using ITensors: ITensor, Index
using TensorOperations
using ITensorNetworks: ITensorNetworks, ITensorNetwork

function had_contract(A::ITensor, B::ITensor, had::Index; α = true)
    @assert NDTensors.datatype(A) == NDTensors.datatype(B)
    dataT = NDTensors.datatype(A)
    if had ∉ commoninds(A, B)
        return α .* (A * B)
    else
        position_of_had_A = findfirst(x -> x == had, inds(A))
        position_of_had_B = findfirst(x -> x == had, inds(B))
        slices_A = eachslice(array(A); dims = position_of_had_A)
        slices_B = eachslice(array(B); dims = position_of_had_B)

        @assert length(slices_A) == length(slices_B)
        inds_c = noncommoninds(A, B)
        elt = promote_type(eltype(A), eltype(B))
        is = vcat(had, inds_c...)
        C = ITensor(dataT(zeros(elt, dim(is))), is)
        ## Right now I have to fill C with zeros because empty tensor
        fill!(C, zero(elt))
        slices_C = eachslice(array(C); dims = 1)
        for i = 1:length(slices_A)
            a_inds = [ind(A, x) for x = 1:ndims(A) if x != position_of_had_A]
            b_inds = [ind(B, x) for x = 1:ndims(B) if x != position_of_had_B]
            ITensors.contract!(
                itensor(slices_C[i], inds_c),
                itensor(slices_A[i], a_inds),
                itensor(slices_B[i], b_inds),
                elt(α),
            )
        end
        return C
    end
end

## TODO this is broken when some items have a rank but others do not.
function had_contract(tensors::Vector{<:ITensor}, had::Index; α = true, sequence = nothing)
    had_tensors = Vector{ITensor}([])
    no_had = Vector{ITensor}([])

    # Find all the tensors that don't contain the hadamard index
    for ten in tensors
        if had ∉ inds(ten)
            push!(no_had, ten)
            continue
        end
        push!(had_tensors, ten)
    end

    # For all that do, find the mode remember its position in the each index list
    positions_of_had = Dict(y => (findfirst(x -> x == had, inds(y))) for y in had_tensors)
    # Slice each tensor along the hadamard dimension
    slices = [eachslice(array(x); dims = positions_of_had[x]) for x in had_tensors]
    slices_inds = [inds(x)[1:end .!= positions_of_had[x]] for x in had_tensors]

    slice_0 = [
        [itensor(slices[x][1], slices_inds[x]) for x = 1:length(had_tensors)]...,
        no_had...,
    ]

    # Find the best way to contract the tensor graph if neessary
    sequence =
        isnothing(sequence) ? ITensors.optimal_contraction_sequence(slice_0) : sequence

    # Contract the full list of tensors but only the first hadamard slice
    cslice = α .* contract(slice_0; sequence)

    # ## Right now I have to fill C with zeros because I hate empty tensor
    # Use the first slice to determine the final indices and make the full resulting tensor
    C = ITensor(zeros(eltype(cslice), dim(had) * dim(cslice)), (had, inds(cslice)...))
    slices_c = eachslice(array(C); dims = 1)
    slices_c[1] .= cslice

    ## TODO would be better to do in place but can't do a list of tensors in place right now
    ## Contract the rest
    for i = 2:dim(had)
        slices_c[i] .= array(
            α .* contract(
                [
                    [
                        itensor(slices[x][i], slices_inds[x]) for x = 1:length(had_tensors)
                    ]...,
                    no_had...,
                ];
                sequence,
            ),
        )
    end

    return C
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
    had_tensors = Vector{ITensor}([])
    no_had = Vector{ITensor}([])

    for ten in tensors
        if had ∉ inds(ten)
            push!(no_had, ten)
            continue
        end
        push!(had_tensors, ten)
    end

    positions_of_had = Dict(y => (findfirst(x -> x == had, inds(y))) for y in had_tensors)
    slices = [eachslice(array(x); dims = positions_of_had[x]) for x in had_tensors]
    slices_inds = [inds(x)[1:end .!= positions_of_had[x]] for x in had_tensors]

    slice_0 = ITensorNetwork([
        [itensor(slices[x][1], slices_inds[x]) for x = 1:length(had_tensors)]...,
        no_had...,
    ])

    sequence =
        isnothing(sequence) ? ITensorNetworks.contraction_sequence(slice_0; alg) : sequence

    cslice = α .* contract(slice_0; sequence)

    C = ITensor(zeros(eltype(cslice), dim(had) * dim(cslice)), (had, inds(cslice)...))

    slices_c = eachslice(array(C); dims = 1)
    slices_c[1] .= cslice

    for i = 2:dim(had)
        slices_c[i] .= array(
            α .* contract(
                ITensorNetwork([
                    [
                        itensor(slices[x][i], slices_inds[x]) for x = 1:length(had_tensors)
                    ]...,
                    no_had...,
                ]);
                sequence,
            ),
        )
    end

    return C
end

function optimal_had_contraction_sequence(tensors::Vector{<:ITensor}, had::Index)
    had_tensors = Vector{ITensor}([])
    no_had = Vector{ITensor}([])

    for ten in tensors
        if had ∉ inds(ten)
            push!(no_had, ten)
            continue
        end
        push!(had_tensors, ten)
    end

    positions_of_had = Dict(y => (findfirst(x -> x == had, inds(y))) for y in had_tensors)
    slices = [eachslice(array(x); dims = positions_of_had[x]) for x in had_tensors]
    slices_inds = [inds(x)[1:end .!= positions_of_had[x]] for x in had_tensors]

    slice_0 = [
        [itensor(slices[x][1], slices_inds[x]) for x = 1:length(had_tensors)]...,
        no_had...,
    ]

    return ITensors.optimal_contraction_sequence(slice_0)
end

function optimal_had_contraction_sequence(tn::ITensorNetwork, had::Index; alg = nothing)
    alg = isnothing(alg) ? "optimal" : alg
    tensors = [tn...]
    had_tensors = Vector{ITensor}([])
    no_had = Vector{ITensor}([])

    for ten in tensors
        if had ∉ inds(ten)
            push!(no_had, ten)
            continue
        end
        push!(had_tensors, ten)
    end

    positions_of_had = Dict(y => (findfirst(x -> x == had, inds(y))) for y in had_tensors)
    slices = [eachslice(array(x); dims = positions_of_had[x]) for x in had_tensors]
    slices_inds = [inds(x)[1:end .!= positions_of_had[x]] for x in had_tensors]

    slice_0 = ITensorNetwork([
        [itensor(slices[x][1], slices_inds[x]) for x = 1:length(had_tensors)]...,
        no_had...,
    ])

    return ITensorNetworks.contraction_sequence(slice_0; alg)
end
