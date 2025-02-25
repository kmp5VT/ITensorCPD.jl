using ITensors
using ITensors.NDTensors: similartype
using Random

struct CPD{TargetT}
    target::TargetT
    factors::Vector{ITensor}
    λ::ITensor
end

# CPD(target, factors, λ) = CPD(target, factors, λ)

factors(cp::CPD) = getproperty(cp, :factors)

Base.getindex(cp::CPD, i) = cp.factors[i]
Base.getindex(cp::CPD) = cp.λ

cp_rank(cp::CPD) = ind(cp[], 1)

function Base.copy(cp::CPD)
    return CPD(cp.target, copy(cp.factors), copy(cp.λ))
end

Base.eltype(cp::CPD) = return eltype(cp.λ)

## This makes a random CPD for a given ITensor
function random_CPD(target::ITensor, rank::Index; rng = nothing)
    rng = isnothing(rng) ? MersenneTwister(3) : rng
    elt = eltype(target)
    cp = Vector{ITensor}([])
    l = nothing

    for i in inds(target)
        rtensor, l = row_norm(random_itensor(rng, elt, rank, i), i)
        push!(cp, rtensor)
    end
    return CPD(target, cp, l)
end

using ITensorNetworks: ITensorNetwork, nv, vertices
function random_CPD(target::ITensorNetwork, rank::Index; rng = nothing)
    rng = isnothing(rng) ? MersenneTwister(3) : rng
    verts = vertices(target)
    elt = eltype(target[first(verts)])
    cp = Vector{ITensor}([])
    partial_mtkrp = similar(cp)
    num_tensors = nv(target)
    external_ind_to_vertex = Dict()
    extern_ind_to_factor = Dict()
    factor_number_to_partial_cont_number = Dict()

    ## What we need to do is loop through every
    ## vertex and find the common/non-common inds.
    ## for every noncommonind push
    factor_number = 1
    partial_cont_number = 1
    for v in verts
        partial = target[v]
        for uniq in uniqueinds(target, v)
            external_ind_to_vertex[uniq] = v
            factor = row_norm(random_itensor(rng, elt, rank, uniq), uniq)[1]
            push!(cp, factor)
            partial = had_contract(partial, factor, rank)
            extern_ind_to_factor[uniq] = factor_number
            factor_number_to_partial_cont_number[factor_number] = partial_cont_number
            factor_number += 1
        end
        push!(partial_mtkrp, partial)
        partial_cont_number += 1
    end

    l = fill!(ITensor(elt, rank), zero(elt))
    return CPD(
        target,
        cp,
        l,
        network_solver(),
        Dict(
            :partial_mtkrp => partial_mtkrp,
            :ext_ind_to_vertex => external_ind_to_vertex,
            :ext_ind_to_factor => extern_ind_to_factor,
            :factor_to_part_cont => factor_number_to_partial_cont_number,
        ),
    )
end
