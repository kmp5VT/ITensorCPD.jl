## This function optimizes the CP-ALS via the normal equations
## [in] cp is a CPD opject that contains the factor matrices to be optimized
## [in] als An ALS object that contains the algorithm which will be used to form the normal equations
## [in] verbose a boolean that determines if the algorithm should print optimization information.
## [out] A CPD object with optmiized ALS factors
function optimize(cp::CPD, als::ALS; verbose = false)
    rank = cp_rank(cp)
    iter = 0

    λ = deepcopy(cp.λ)
    factors = deepcopy(cp.factors)
    num_factors = length(cp.factors)
    
    converge = als.check
    while iter < converge.max_counter
        mtkrp = nothing
        for (fact, target_ind) in zip(1:num_factors, inds(cp))
            ## compute the matrized tensor time khatri rao product with a provided algorithm.
            krp = compute_krp(als.mttkrp_alg, als, factors, cp, rank, fact)

            mtkrp = matricize_tensor(als.mttkrp_alg, als, factors, cp, rank, fact)

            solution = solve_ls_problem(als.mttkrp_alg, als, krp, mtkrp, rank)
            
            factors[fact], λ = row_norm(solution, target_ind)

            post_solve(als.mttkrp_alg, als, factors, λ, cp, rank, fact)
        end

        if check_converge(als.mttkrp_alg, converge, als, mtkrp, factors, λ, verbose) && break end
        iter += 1
    end

    return CPD{typeof(als.target)}(factors, λ)
end
