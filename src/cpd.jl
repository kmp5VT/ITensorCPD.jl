using ITensors
using ITensors: Indices
using ITensors.NDTensors: similartype
using Random

struct CPD{TargetT}
    factors::Vector{ITensor}
    λ::ITensor
    inds::Vector{Index}
end

function CPD{TargetT}(factors::Vector{ITensor}, λ::ITensor) where {TargetT}
    is = [ind(x, 1) for x in factors]
    CPD{TargetT}(factors, λ, is)
end

factors(cp::CPD) = getproperty(cp, :factors)
ITensors.inds(cp::CPD) = getproperty(cp, :inds)
ITensors.ind(cp::CPD, i::Int) = inds(cp)[i]
ITensors.itensor2inds(A::CPD)::Any = inds(A)
paramT(cp::CPD{T}) where {T} = T
Base.length(cp::CPD) = length(cp.factors)

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

function random_factors(elt::Union{<:Type, Nothing}, is::Indices, rank::Index; rng=nothing)
    @assert !isnothing(elt)
    rng = isnothing(rng) ? MersenneTwister(3) : rng
    cp = Vector{ITensor}([])
    l = nothing
    for i in is
        it = random_itensor(rng, elt, i, rank)
        rtensor, l = row_norm(it, i)
        push!(cp, rtensor)
    end

    return cp, l
end

## This makes a random CPD for a given ITensor
function random_CPD(target::ITensor, rank::Index; rng = nothing)
    factors, lambda = random_factors(eltype(target), inds(target), rank; rng)
    return CPD{ITensor}(factors, lambda)
end

## TODO add a random hash key to label
function random_CPD(target, rank::Int; rng = nothing)
    return random_CPD(target, ITensors.Index(rank, "CPD"); rng)
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
    is = vcat([uniqueinds(target, v) for v in verts]...)
    factors, lambda = random_factors(elt, is, rank; rng);

    return CPD{ITensorNetwork}(factors, lambda)
end
