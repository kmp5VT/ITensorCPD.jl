using LinearAlgebra: ColumnNorm, diagm
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
    extra_args[:target_transform] = [target for i in inds(target)]
    return optimize_diff_projection(cp, ALS(target, alg, extra_args, check); verbose)
end

function als_optimize(
    alg::DoubleInterp,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    verbose = false,
)
    projectors = Vector{Vector{Int}}()
    targets = Vector{ITensor}()
    for i in inds(target)
        Ris = uniqueinds(target, i)
        Tmat = reshape(array(target, (i, Ris...)), (dim(i), dim(Ris)))
        _, _, p = qr(Tmat, ColumnNorm())
        push!(projectors, p)

        dRis = dim(Ris)
        int_end = stop(alg)
        int_end = iszero(int_end) ? dRis : int_end
        int_end = dRis < int_end ? dRis : int_end

        int_start = start(alg)
        @assert int_start > 0 && int_start ≤ int_end

        ndim = int_end - int_start + 1
        t = zeros(eltype(Tmat), (dRis, ndim))
        j = 1
        for i = int_start:int_end
            t[p[i], j] = 1
            j += 1
        end
        piv_id = Index(ndim, "pivot")
        push!(targets, itensor(t, Ris, piv_id) * itensor(t, Ris', piv_id));
    end
    extra_args[:projects] = projectors
    extra_args[:projects_tensors] = targets
    extra_args[:target_transform] = [noprime(target * x) for x in targets]
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
    alg::InterpolateTarget,
    target::ITensor,
    cp::CPD{<:ITensor};
    extra_args = Dict(),
    check = nothing,
    verbose = false,
)
    projectors = Vector{Vector{Int}}()
    targets = Vector{ITensor}()
    for i in inds(target)
        Ris = uniqueinds(target, i)
        Tmat = reshape(array(target, (i, Ris...)), (dim(i), dim(Ris)))
        _, _, p = qr(Tmat, ColumnNorm())
        push!(projectors, p)

        dRis = dim(Ris)
        int_end = stop(alg)
        int_end = iszero(int_end) ? dRis : int_end
        int_end = dRis < int_end ? dRis : int_end

        int_start = start(alg)
        @assert int_start > 0 && int_start ≤ int_end

        ndim = int_end - int_start + 1
        t = zeros(eltype(Tmat), (dRis, ndim))
        j = 1
        for i = int_start:int_end
            t[p[i], j] = 1
            j += 1
        end
        push!(targets, itensor(t, Ris, Index(ndim, "pivot")));
    end
    extra_args[:target_projects] = projectors
    extra_args[:target_transform] = targets

    # return ALS(target, alg, extra_args, check)
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
            factor_pos = findfirst(x -> x == uniq, ind.(cp.factors, 2))
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
            factors[fact], λ = row_norm(
                itensor(qr(array(dag(grammian)), ColumnNorm()) \ array(mtkrp), inds(mtkrp)),
                ind(mtkrp, 2),
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

    num_factors = length(cp.factors)
    λ = copy(cp.λ)
    factors = copy(cp.factors)
    part_grammian = cp.factors .* dag.(prime.(cp.factors; tags = tags(rank)))

    converge = als.check
    target_inds = inds(als.target)

    while iter < converge.max_counter
        mtkrp = nothing
        for fact = 1:num_factors
            target_ind = target_inds[fact]
            # ### Trying to solve T V = I [(J x K) V] 
            # #### This is the first KRP * Singular values of T: [(J x K) V]  
            factor_portion = factors[1:end .!= fact]
            projected_KRP = project_krp(als.mttkrp_alg, als, factor_portion, cp, rank, fact)
            projected_target =
                project_target(als.mttkrp_alg, als, factor_portion, cp, rank, fact)

            # mtkrp = projected_KRP * als.additional_items[:target_transform][fact];
            # ##### Now contract TV by the inverse of KRP * SVD
            U, S, V = svd(projected_KRP, rank; use_absolute_cutoff = true, cutoff = 0)
            direction = (U * (prime(projected_target; tags = tags(rank)) * V * (1 ./ S)))

            factors[fact], λ = row_norm(direction, target_ind)
            # part_grammian[fact] =
            #     factors[fact] * dag(prime(factors[fact]; tags = tags(rank)))
        end


        # potentially save the MTTKRP for the loss function

        # save_mttkrp(converge, mtkrp)

        recon = reconstruct(factors, λ)
        diff = als.target - recon
        println("Accuracy: $(1.0 - norm(diff) / norm(als.target))")
        # if check_converge(converge, factors, λ, part_grammian; verbose)
        #     break
        # end
        iter += 1
    end

    return CPD{typeof(als.target)}(factors, λ)
end
