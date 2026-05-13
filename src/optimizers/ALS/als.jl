using LinearAlgebra: ColumnNorm, diagm
using ITensors.NDTensors:Diag
using ITensors: tags

struct ALS <: CPDOptimizer
    target::Any
    mttkrp_alg::Union{MttkrpAlgorithm,ProjectionAlgorithm}
    additional_items::Dict
    check::ConvergeAlg
end

Base.copy(als::ALS) = ALS(als.target, als.mttkrp_alg, copy(als.additional_items), als.check)
iter(als::ALS) = iter(als.check)

function als_optimize(
    target,
    cp::CPD;
    alg = nothing,
    check = nothing,
    maxiter = nothing,
    verbose = false,
    kwargs...)
    als = compute_als(target, cp; alg, check, maxiter, kwargs...)
    optimize(cp, als; verbose)
end

include("standard/tensor.jl")
include("standard/network.jl")

include("randomized/qr_lev_score_sampled.jl")
include("randomized/krp_lev_score_sampled.jl")
include("randomized/sketched_ls.jl")

### Default ALS constructor algorithm for Tensors (versus tensor networks). 
### This will develop the "optimization sequence" variable
### and then pass along to more specialized constructors
function compute_als(
    target::ITensor,
    cp::CPD{<:ITensor};
    alg = nothing,
    check = nothing,
    maxiter = nothing,
    kwargs...
)
    alg = isnothing(alg) ? KRPFreeNormal() : alg
    extra_args = Dict();
    check = isnothing(check) ? NoCheck(isnothing(maxiter) ? 100 : maxiter) : check
    mttkrp_contract_sequences = Vector{Union{Any,Nothing}}()
    permute_symm_inds = haskey(kwargs, :permute_symm_inds) ? kwargs[:permute_symm_inds] : [1:length(inds(target))...]
    @assert length(permute_symm_inds) == length(inds(target))
    for l in inds(target)
        push!(mttkrp_contract_sequences, nothing)
    end
    extra_args[:mttkrp_contract_sequences] = mttkrp_contract_sequences
    extra_args[:permute_symm_inds] = permute_symm_inds
    cprank = cp_rank(cp)
    return compute_als(alg, target, cp; extra_args, check, kwargs...)
end

### Default constructor algorithms for randomized ALS solvers (ProjectionAlgorithm).
function compute_als(
    alg::ProjectionAlgorithm,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    normal = false,
    kwargs...
)
    extra_args[:normal] = normal
    return ALS(target, alg, extra_args, check)
end