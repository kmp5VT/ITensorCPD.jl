using LinearAlgebra: ColumnNorm, diagm
using ITensors.NDTensors:Diag
abstract type CPDOptimizer end

struct ALS <: CPDOptimizer
    target::Any
    mttkrp_alg::Union{MttkrpAlgorithm,ProjectionAlgorithm}
    additional_items::Dict
    check::ConvergeAlg
end

function als_optimize(
    target::ITensor,
    cp::CPD{<:ITensor};
    alg = nothing,
    check = nothing,
    maxiter = nothing,
    verbose = false,
)
    alg = isnothing(alg) ? direct() : alg
    extra_args = Dict();
    check = isnothing(check) ? NoCheck(isnothing(maxiter) ? 100 : maxiter) : check
    mttkrp_contract_sequences = Vector{Union{Any,Nothing}}()
    for l in inds(target)
        push!(mttkrp_contract_sequences, nothing)
    end
    extra_args[:mttkrp_contract_sequences] = mttkrp_contract_sequences
    als_optimize(alg, target, cp; extra_args, check, verbose)
end

function als_optimize(
    alg::MttkrpAlgorithm,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    verbose = false,
)
    return optimize(cp, ALS(target, alg, extra_args, check); verbose)
end

function als_optimize(
    alg::InvKRP,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    verbose = false,
)
    return optimize_diff_projection(cp, ALS(target, alg, extra_args, check); verbose)
end

function als_optimize(
    alg::QRPivProjected,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    verbose = false,
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
    
    # return ALS(target, alg, extra_args, check)
    return optimize_diff_projection(cp, ALS(target, alg, extra_args, check); verbose)
end

function als_optimize(
    alg::LevScoreSampled,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    verbose = false,
)
    # pivots = Vector{Vector{Int}}()
    # projectors = Vector{ITensor}()

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
    # return ALS(target, alg, extra_args, check)
    return optimize_diff_projection(cp, ALS(target, alg, extra_args, check); verbose)
end

function als_optimize(
    alg::TargetDecomp,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    verbose = false,
)
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

    return optimize(cp, ALS(target, alg, extra_args, check); verbose)
end

function als_optimize(
    target::ITensorNetwork,
    cp::CPD{<:ITensorNetwork};
    alg = nothing,
    check = nothing,
    maxiter = nothing,
    verbose = false,
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
            :partial_mtkrp => partial_mtkrp,
            :ext_ind_to_vertex => external_ind_to_vertex,
            :ext_ind_to_factor => extern_ind_to_factor,
            :factor_to_part_cont => factor_number_to_partial_cont_number,
            :mttkrp_contract_sequences => mttkrp_contract_sequences,
        ),
        check,
    )
    optimize(cp, als; verbose)
end

function optimize(cp::CPD, als::ALS; verbose = true)
    rank = cp_rank(cp)
    iter = 0
    part_grammian = cp.factors .* dag.(prime.(cp.factors; tags = tags(rank)))
    num_factors = length(cp.factors)
    λ = copy(cp.λ)
    factors = copy(cp.factors)
    converge = als.check
    while iter < converge.max_counter
        mtkrp = nothing
        for fact = 1:num_factors
            ## compute the matrized tensor time khatri rao product with a provided algorithm.
            mtkrp = mttkrp(als.mttkrp_alg, als, factors, cp, rank, fact)

            ## compute the grammian which requires the hadamard product
            grammian = similar(part_grammian[1])
            fill!(grammian, one(eltype(cp)))
            for i = 1:num_factors
                if i == fact
                    continue
                end
                grammian = hadamard_product(grammian, part_grammian[i])
            end

            ## potentially better to first inverse the grammian then contract
            ## qr(A, Val(true))
            solution = qr(array(dag(grammian)), ColumnNorm()) \ transpose(array(mtkrp))
            
            factors[fact], λ = row_norm(
                itensor(copy(transpose(solution)), inds(mtkrp)),
                ind(cp, fact),
            )
            part_grammian[fact] =
                factors[fact] * dag(prime(factors[fact]; tags = tags(rank)))

            post_solve(als.mttkrp_alg, als, factors, λ, cp, rank, fact)
        end

        # potentially save the MTTKRP for the loss function

        save_mttkrp(converge, mtkrp)

        if check_converge(converge, factors, λ, part_grammian; verbose)
            break
        end
        iter += 1
    end

    return CPD{typeof(als.target)}(factors, λ)
end

function optimize_diff_projection(cp::CPD, als::ALS; verbose = true)
    rank = cp_rank(cp)
    iter = 0

    λ = copy(cp.λ)
    factors = copy(cp.factors)

    converge = als.check
    target_inds = inds(als.target)

    while iter < converge.max_counter
        for fact = 1:length(cp)
            target_ind = target_inds[fact]
            # ### Trying to solve T V = I [(J x K) V] 
            # #### This is the first KRP * Singular values of T: [(J x K) V]  
            factor_portion = factors[1:end .!= fact]
            projected_KRP = project_krp(als.mttkrp_alg, als, factor_portion, cp, rank, fact)
            
            projected_target =
                project_target(als.mttkrp_alg, als, factor_portion, cp, rank, fact, projected_KRP)

            # ##### Now contract TV by the inverse of KRP * SVD
            direction = solve_ls_problem(als.mttkrp_alg, projected_KRP, projected_target, rank)

            factors[fact], λ = row_norm(direction, target_ind)

            post_solve(als.mttkrp_alg, als, factors, λ, cp, rank, fact)
        end

        recon = reconstruct(factors, λ)
        diff = als.target - recon
        println("Accuracy: $(1.0 - norm(diff) / norm(als.target))")
        iter += 1
    end

    return CPD{typeof(als.target)}(factors, λ)
end
