## This function optimizes the CP-ALS via the normal equations
## [in] cp is a CPD opject that contains the factor matrices to be optimized
## [in] als An ALS object that contains the algorithm which will be used to form the normal equations
## [in] verbose a boolean that determines if the algorithm should print optimization information.
## [out] A CPD object with optmiized ALS factors
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

## This function optimizes the CP-ALS via a projected LS equation i.e. |PT_a - A (B ⊙ C) P |²
## This function does not expilcitly form the normal equation just computes the projected and matricized T, the 
## projected khatri rao product and solves the linear equation PT_a = A (B ⊙ C)P.
## Note the solve_ls_problem function can be modified to form the normal equation based solution.

## [in] cp is a CPD opject that contains the factor matrices to be optimized
## [in] als An ALS object that contains the algorithm which will be used to form the normal equations
## [in] verbose a boolean that determines if the algorithm should print optimization information.
## [out] A CPD object with optmiized ALS factors
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

        if als.check isa FitCheck && verbose
            inner_prod = (ITensorCPD.had_contract([als.target, factors...], rank) * λ)[]
            partial_gram = [fact * dag(prime(fact; tags=tags(rank))) for fact in factors];
            fact_square = ITensorCPD.norm_factors(partial_gram, λ)
            normResidual =
                sqrt(abs(als.check.ref_norm * als.check.ref_norm + fact_square - 2 * abs(inner_prod)))
            println("Accuracy: $(1.0 - normResidual / norm(als.check.ref_norm))")
        end
        iter += 1
    end

    return CPD{typeof(als.target)}(factors, λ)
end

