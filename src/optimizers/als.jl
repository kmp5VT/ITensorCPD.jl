using LinearAlgebra: ColumnNorm, diagm
abstract type CPDOptimizer end

struct ALS <: CPDOptimizer
    target
    mttkrp_alg::MttkrpAlgorithm
    additional_items::Dict
    check::ConvergeAlg
end

function als_optimize(target::ITensor, cp::CPD{<:ITensor}; alg=nothing, check=nothing, maxiter=nothing, verbose=false)
    alg = isnothing(alg) ? direct() : alg
    check = isnothing(check) ? NoCheck(isnothing(maxiter) ? 100 : maxiter) : check
    return optimize(cp, ALS(target, alg, Dict(), check); verbose)
end

function als_optimize(target::ITensorNetwork, cp::CPD{<:ITensorNetwork}; alg=network_solver(), 
    check=nothing, maxiter=nothing, verbose=false)
    check = isnothing(check) ? NoCheck(isnothing(maxiter) ? 100 : maxiter) : check
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
            factor = cp.factors[factor_pos]
            partial = had_contract(partial, factor, cpRank)
            extern_ind_to_factor[uniq] = factor_pos
            factor_number_to_partial_cont_number[factor_pos] = partial_cont_number
            factor_number += 1
        end
        push!(partial_mtkrp, partial)
        partial_cont_number += 1
    end

    als = ALS(target, 
    alg,
    Dict(
        :partial_mtkrp => partial_mtkrp,
        :ext_ind_to_vertex => external_ind_to_vertex,
        :ext_ind_to_factor => extern_ind_to_factor,
        :factor_to_part_cont => factor_number_to_partial_cont_number,
        ),
        check
        )
    optimize(cp, als; verbose)
end

function optimize(cp::CPD, als::ALS; verbose=true)
    rank = cp_rank(cp)
    iter = 0
    part_grammian = cp.factors .* prime.(cp.factors; tags = tags(rank))
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
                itensor(qr(array(grammian), ColumnNorm()) \ array(mtkrp), inds(mtkrp)),
                ind(mtkrp, 2),
            )
            part_grammian[fact] = factors[fact] * prime(factors[fact]; tags = tags(rank))

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

# function als_direct_optimize(cp::CPD, als::ALS, converge)
#     rank = cp_rank(cp)
#     iter = 0
#     part_grammian = cp.factors .* prime.(cp.factors; tags = tags(rank))
#     num_factors = length(cp.factors)
#     λ = copy(cp.λ)
#     factors = copy(cp.factors)
#     while iter < converge.max_counter
#         mtkrp = nothing
#         for fact = 1:num_factors
#             ## compute the matrized tensor time khatri rao product with a provided algorithm.
#             mtkrp = mttkrp(als.mttkrp_alg, factors, cp, rank, fact)

#             ## compute the grammian which requires the hadamard product
#             grammian = similar(part_grammian[1])
#             fill!(grammian, one(eltype(cp)))
#             for i = 1:num_factors
#                 if i == fact
#                     continue
#                 end
#                 grammian = hadamard_product(grammian, part_grammian[i])
#             end

#             w = had_contract(factors[1:end.!=fact], rank)
#             col_dim = prod(size(w)[2:end])
#             ## potentially better to first inverse the grammian then contract
#             ## qr(A, Val(true))
#             winv = pseudoinverse(w, inds(w)[2:end])
#             factors[fact], λ = row_norm(winv * cp.target, ind(mtkrp, 2))
#             part_grammian[fact] = factors[fact] * prime(factors[fact]; tags = tags(rank))

#             post_solve(cp.mttkrp_alg, factors, λ, cp, rank, fact)
#         end

#         # potentially save the MTTKRP for the loss function
#         save_mttkrp(converge, mtkrp)

#         if check_converge(converge, factors, λ, part_grammian)
#             break
#         end
#         iter += 1
#     end

#     return CPD(cp.target, factors, λ, cp.mttkrp_alg, cp.additional_items)
# end
