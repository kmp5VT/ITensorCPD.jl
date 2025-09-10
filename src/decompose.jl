function decompose(A, rank::Int; KWARGS...)
    decompose(A, Index(rank, "CP rank"); KWARGS...)
end

function decompose(
    A,
    rank_ind::Index;
    solver::Union{CPDOptimizer,Nothing} = nothing,
    rng = nothing,
    alg = nothing,
    check = nothing,
    maxiter = nothing,
    verbose = false,
)
    CP = random_CPD(A, rank_ind; rng)
    if isnothing(solver)
        return als_optimize(A, CP; alg, check, maxiter, verbose)
    else
        throw("OptimizerError")
    end
end

function decompose(
    A,
    epsilon,
    max_rank::Int;
    rng = nothing,
    alg = nothing,
    check=nothing,
    maxiter = nothing,
    verbose=false,
    start_rank = 1,
    rank_step = 1,
)
    if verbose
        println("Starting with rank: $(start_rank)")
    end
    current_rank = start_rank
    cpd = random_CPD(A, start_rank; rng)
    check = isnothing(check) ? FitCheck(1e-3, 100, norm(A)) : check
    while true
        cpd = als_optimize(A, cpd; alg, check, maxiter, verbose);
        if 1.0 - ITensorCPD.CPDFit(check) < epsilon
            return cpd
        else
            current_rank += rank_step;
            verbose && println("\nIncreasing rank to: $(current_rank)")    
        end

        if current_rank > max_rank
            println("Optimization Failed to converge within rank $(max_rank)")
            return cpd
        end
        
        cpd = increase_cpd_rank(cpd, Index(current_rank, tags(cp_rank(cpd))); rng)
    end
    return cpd
end

function increase_cpd_rank(cpd::ITensorCPD.CPD, new_rank::Index; rng = nothing)
    rng = isnothing(rng) ? MersenneTwister(3) : rng
    elt = eltype(cpd)

    old_rank = cp_rank(cpd)
    @assert dim(new_rank) ≥ dim(old_rank)

    updated_factors, lambda = random_factors(eltype(cpd), inds(cpd), new_rank; rng)
    for (old, new) in zip(cpd.factors, updated_factors)
        array(new)[:,1:dim(old_rank)] = array(old)
    end
    return ITensorCPD.CPD{paramT(cpd)}(updated_factors, lambda)
end