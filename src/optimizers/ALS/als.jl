using LinearAlgebra: ColumnNorm, diagm
using ITensors.NDTensors:Diag
using ITensors: tags
abstract type CPDOptimizer end

struct ALS <: CPDOptimizer
    target::Any
    mttkrp_alg::Union{MttkrpAlgorithm,ProjectionAlgorithm}
    additional_items::Dict
    check::ConvergeAlg
end

include("optimize.jl")

function als_optimize(
    target,
    cp::CPD;
    alg = nothing,
    check = nothing,
    maxiter = nothing,
    verbose = false,)
    als = compute_als(target, cp; alg, check, maxiter)
    optimize(cp, als; verbose)
end

## Default ALS constructor algorithm for Tensors (versus tensor networks). 
## This will develop the "optimization sequence" variable
## and then pass along to more specialized constructors
function compute_als(
    target::ITensor,
    cp::CPD{<:ITensor};
    alg = nothing,
    check = nothing,
    maxiter = nothing
)
    alg = isnothing(alg) ? direct() : alg
    extra_args = Dict();
    check = isnothing(check) ? NoCheck(isnothing(maxiter) ? 100 : maxiter) : check
    mttkrp_contract_sequences = Vector{Union{Any,Nothing}}()
    for l in inds(target)
        push!(mttkrp_contract_sequences, nothing)
    end
    extra_args[:mttkrp_contract_sequences] = mttkrp_contract_sequences
    cprank = cp_rank(cp)
    return compute_als(alg, target, cp; extra_args, check)
end

## Default constructor algorithms for normal equation based solvers (MttkrpAlgorithm).
## This needs no extra information and passes the function `optimize` as the optimizer algorithm.
function compute_als(
    alg::MttkrpAlgorithm,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
)
    cprank = cp_rank(cp)
    extra_args[:part_grammian] = cp.factors .* dag.(prime.(cp.factors; tags = tags(cprank)))
    return ALS(target, alg, extra_args, check)
end

function compute_als(
    alg::TargetDecomp,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
)
    cprank = cp_rank(cp)
    extra_args[:part_grammian] = cp.factors .* dag.(prime.(cp.factors; tags = tags(cprank)))
    decomps = Vector{ITensor}()
    targets = Vector{ITensor}()
    for i in inds(target)
        u, s, v = svd(target, i; use_relative_cutoff = false, cutoff = 0)
        push!(decomps, v)
        t = u * s
        push!(targets, t);
    end
    extra_args[:target_decomps] = decomps
    extra_args[:target_transform] = targets

    return ALS(target, alg, extra_args, check)
end

function compute_als(
    target::ITensorNetwork,
    cp::CPD{<:ITensorNetwork};
    alg = nothing,
    check = nothing,
    maxiter = nothing,
)
    check = isnothing(check) ? NoCheck(isnothing(maxiter) ? 100 : maxiter) : check
    alg = isnothing(alg) ? network_solver() : alg
    verts = vertices(target)
    elt = eltype(target[first(verts)])
    cpRank = cp_rank(cp)

    partial_mtkrp = typeof(similar(cp.factors))()
    external_ind_to_vertex = Dict()
    extern_ind_to_factor = Dict()
    factor_number_to_partial_cont_number = Dict()

    # What we need to do here is walk through the CP and find where
    # Each factor connects to in the given target newtork.
    # This will assume all external legs are connected.
    factor_number = 1
    partial_cont_number = 1
    for v in verts
        partial = target[v]
        for uniq in uniqueinds(target, v)
            external_ind_to_vertex[uniq] = v
            factor_pos = findfirst(x -> x == uniq, inds(cp))
            factor = dag(cp.factors[factor_pos])
            partial = had_contract(partial, factor, cpRank)
            extern_ind_to_factor[uniq] = factor_pos
            factor_number_to_partial_cont_number[factor_pos] = partial_cont_number
            factor_number += 1
        end
        push!(partial_mtkrp, partial)
        partial_cont_number += 1
    end

    mttkrp_contract_sequences = Vector{Union{Any,Nothing}}()
    for _ in inds(cp)
        push!(mttkrp_contract_sequences, nothing)
    end

    als = ALS(
        target,
        alg,
        Dict(
            :part_grammian => cp.factors .* dag.(prime.(cp.factors; tags = tags(cpRank))),
            :partial_mtkrp => partial_mtkrp,
            :ext_ind_to_vertex => external_ind_to_vertex,
            :ext_ind_to_factor => extern_ind_to_factor,
            :factor_to_part_cont => factor_number_to_partial_cont_number,
            :mttkrp_contract_sequences => mttkrp_contract_sequences,
        ),
        check,
    )
    return als
end

## Default constructor algorithms for non-normal equation based solvers (ProjectionAlgorithm).
## This needs no extra information and passes the function `optimize` as the optimizer algorithm.
function compute_als(
    alg::ProjectionAlgorithm,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
)
    return ALS(target, alg, extra_args, check)
end

function compute_als(
    alg::QRPivProjected,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
)
    pivots = Vector{Vector{Int}}()
    projectors = Vector{ITensor}()
    targets = Vector{ITensor}()
    piv_id = nothing
    for (i, n) in zip(inds(target), 1:length(cp))
        Ris = uniqueinds(target, i)
        Tmat = reshape(array(target, (i, Ris...)), (dim(i), dim(Ris)))
        _, _, p = qr(Tmat, ColumnNorm())
        push!(pivots, p)

        dRis = dim(Ris)
        int_end = stop(alg)
        int_end = length(int_end) == 1 ? int_end[1] : int_end[n]
        int_end = iszero(int_end) ? dRis : int_end
        int_end = dRis < int_end ? dRis : int_end

        int_start = start(alg)
        int_start = length(int_start) == 1 ? int_start[1] : int_start[n]
        @assert int_start > 0 && int_start ≤ int_end

        ndim = int_end - int_start + 1
        piv_id = Index(ndim, "pivot")

        push!(projectors, itensor(tensor(Diag(p[int_start:int_end]), (Ris..., piv_id))))
        TP = fused_flatten_sample(target, n, projectors[n])
    push!(targets, TP)
    end
    extra_args[:projects] = pivots
    extra_args[:projects_tensors] = projectors
    extra_args[:target_transform] = targets
    
    return ALS(target, alg, extra_args, check)
end

function compute_als(
    alg::SEQRCSPivProjected,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
)
    lst = random_modes(alg)
    lst = isnothing(lst) ? [] : lst
    rank_sk = rank_vect(alg)
    pivots = Vector{Vector{Int}}()
    projectors = Vector{ITensor}()
    targets = Vector{ITensor}()
    piv_id = nothing
    for (i, n) in zip(inds(target), 1:length(cp))
        Ris = uniqueinds(target, i)

        dRis = dim(Ris)
        int_end = stop(alg)
        int_end = length(int_end) == 1 ? int_end[1] : int_end[n]
        int_end = iszero(int_end) ? dRis : int_end
        int_end = dRis < int_end ? dRis : int_end

        int_start = start(alg)
        int_start = length(int_start) == 1 ? int_start[1] : int_start[n]
        @assert int_start > 0 && int_start ≤ int_end

        # TODO Use the randomized linear algebra to remove the need to form the matricized tensor.
        if n in lst
            k_sk = isnothing(rank_sk) ? int_end[n] : rank_sk[n]
            m = dim(i)
            l=Int(round(3 * m * log(m))) 
            s=Int(round(log(m)))
             _,_,p = SEQRCS(target,n,i,l,s,k_sk)
        else
            Tmat = reshape(array(target, (i, Ris...)), (dim(i), dim(Ris)))
            _, _, p = qr(Tmat, ColumnNorm())
        end
        push!(pivots, p)

        ndim = int_end - int_start + 1
        piv_id = Index(ndim, "pivot")

        push!(projectors, itensor(tensor(Diag(p[int_start:int_end]), (Ris..., piv_id))))
        TP = fused_flatten_sample(target, n, projectors[n])
    push!(targets, TP)
    end
    extra_args[:projects] = pivots
    extra_args[:projects_tensors] = projectors
    extra_args[:target_transform] = targets
    
    return ALS(target, alg, extra_args, check)
end

function compute_als(
    alg::LevScoreSampled,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
)
    ## For each factor matrix compute its weights
    extra_args[:factor_weights] = [compute_leverage_score_probabilitiy(cp[i], ind(cp, i)) for i in 1:length(cp)]
    pivot_tensors = Vector{ITensor}()
    for fact in 1:length(cp)
        ## grab the tensor indices for all other factors but fact
        Ris = inds(cp)[1:end .!= fact]
        dRis = dims(Ris)

        ## sample the factor matrix space one taking nsamps independent samples from each
        ## tensor
        nsamps = nsamples(alg)
        nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]
        sampled_cols = sample_factor_matrices(nsamps, fact, extra_args[:factor_weights])

        ## We store the rows of the flattened tensor (i.e. α) because it can be converted to factor values
        ## or values in the vectorized tensor
        sampled_tensor_cols = multi_coords_to_column(dRis, sampled_cols)
        piv_ind = Index(length(sampled_tensor_cols), "selector_$(fact)")

        ## make the canonical pivot tensor. This list of pivots will be overwritten each ALS iteration
        push!(pivot_tensors, itensor(tensor(Diag(sampled_tensor_cols), (Ris..., piv_ind))))
    end
    extra_args[:pivot_tensors] = pivot_tensors
    return ALS(target, alg, extra_args, check)
end

function compute_als(
    alg::BlockLevScoreSampled,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
)
    ## For each factor matrix compute its weights
    extra_args[:factor_weights] = [compute_leverage_score_probabilitiy(cp[i], ind(cp, i)) for i in 1:length(cp)]
    pivot_tensors = Vector{ITensor}()
    for fact in 1:length(cp)
        ## grab the tensor indices for all other factors but fact
        Ris = inds(cp)[1:end .!= fact]
        dRis = dims(Ris)

        ## sample the factor matrix space one taking nsamps independent samples from each
        ## tensor
        nsamps = nsamples(alg)
        nsamps = length(nsamps) == 1 ? nsamps[1] : nsamps[fact]
        block_size = blocks(alg)
        block_size = length(block_size) == 1 ? block_size[1] : block_size[fact]
        
        sampled_cols = block_sample_factor_matrices(nsamps, extra_args[:factor_weights] , block_size, fact)

        ## We store the rows of the flattened tensor (i.e. α) because it can be converted to factor values
        ## or values in the vectorized tensor
        sampled_tensor_cols = multi_coords_to_column(dRis, sampled_cols)
        piv_ind = Index(length(sampled_tensor_cols), "selector_$(fact)")

        ## make the canonical pivot tensor. This list of pivots will be overwritten each ALS iteration
        push!(pivot_tensors, itensor(tensor(Diag(sampled_tensor_cols), (Ris..., piv_ind))))
    end
    ## Notice the pivot tensor is actually a low rank tensor it stores the diagonal pivot values
    ## in α form (rows of the matricized tensor) and the indices which are captured in the pivot.
    ## The order of indices are (indices which connect to the pivot, pivot_index).
    extra_args[:pivot_tensors] = pivot_tensors
    return ALS(target, alg, extra_args, check)
end
