using ITensors
using ITensors.NDTensors: similartype
using Random

struct CPD{TargetT}
    factors::Vector{ITensor}
    λ::ITensor
    inds::Vector{Index}
end

function CPD{TargetT}(factors::Vector{ITensor}, λ::ITensor) where {TargetT}
    is = [ind(x, 2) for x in factors]
    CPD{TargetT}(factors, λ, is)
end

# CPD(target, factors, λ) = CPD(target, factors, λ)

factors(cp::CPD) = getproperty(cp, :factors)
ITensors.inds(cp::CPD) = getproperty(cp, :inds)

Base.getindex(cp::CPD, i) = cp.factors[i]
Base.getindex(cp::CPD) = cp.λ

function Base.isequal(cp1::CPD, cp2::CPD)
    cp1.factors == cp2.factors && cp1.λ == cp2.λ
end

cp_rank(cp::CPD) = ind(cp[], 1)

function Base.copy(cp::CPD{T}) where {T}
    return CPD{T}(copy(cp.factors), copy(cp.λ))
end

Base.eltype(cp::CPD) = return eltype(cp.λ)

## This makes a random CPD for a given ITensor
function random_CPD(target::ITensor, rank::Index; rng = nothing)
    rng = isnothing(rng) ? MersenneTwister(3) : rng
    cp = Vector{ITensor}([])
    l = nothing

    for i in inds(target)
        it = itensor(NDTensors.randomTensor(NDTensors.datatype(target), (rank, i)))
        rtensor, l = row_norm(it, i)
        push!(cp, rtensor)
    end
    return CPD{ITensor}(cp, l)
end

function random_CPD(target, rank::Int; rng = nothing)
    random_CPD(target, Index(rank, "CPD"); rng)
end

using ITensorNetworks: ITensorNetwork, nv, vertices
function random_CPD(target::ITensorNetwork, rank::Index; rng = nothing)
    rng = isnothing(rng) ? MersenneTwister(3) : rng
    verts = vertices(target)
    elt = eltype(target[first(verts)])
    cp = Vector{ITensor}([])

    ## What we need to do is loop through every
    ## vertex and find the common/non-common inds.
    ## for every noncommonind push
    for v in verts
        for uniq in uniqueinds(target, v)
            factor = row_norm(random_itensor(rng, elt, rank, uniq), uniq)[1]
            push!(cp, factor)
        end
    end

    l = fill!(ITensor(elt, rank), zero(elt))
    return CPD{ITensorNetwork}(cp, l)
end
