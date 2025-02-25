using LinearAlgebra: ColumnNorm, diagm
abstract type CPDOptimizer end

struct ALS <: CPDOptimizer
    mttkrp_alg::MttkrpAlgorithm
    additional_items::Dict
    check::ConvergeAlg
end

function decompose(A, rank::Int; solver::Union{CPDOptimizer,Nothing} = nothing, rng=nothing, alg=nothing, check=nothing, maxiter=nothing, verbose=false)
    CP = random_CPD(A, Index(rank, "CP Rank"); rng)
    if isnothing(solver)
        return als_optimize(CP; alg, check, maxiter, verbose)
    else
        throw("OptimizerError")
    end
end

function als_optimize(cp::CPD{<:ITensor}; alg=nothing, check=nothing, maxiter=nothing, verbose=false)
    alg = isnothing(alg) ? direct() : alg
    check = isnothing(check) ? NoCheck(isnothing(maxiter) ? 100 : maxiter) : check
    return als_optimize(cp, ALS(alg, Dict(), check); verbose)
end

function als_optimize(cp::CPD{<:ITensorNetwork})
end

function als_optimize(cp::CPD, als::ALS; verbose=true)
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
            mtkrp = mttkrp(als.mttkrp_alg, factors, cp, rank, fact)

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

            post_solve(als.mttkrp_alg, factors, λ, cp, rank, fact)
        end

        # potentially save the MTTKRP for the loss function

        save_mttkrp(converge, mtkrp)

        if check_converge(converge, factors, λ, part_grammian; verbose)
            break
        end
        iter += 1
    end

    return CPD(cp.target, factors, λ)
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
