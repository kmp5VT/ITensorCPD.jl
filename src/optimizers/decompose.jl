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

function ITensorCPD.decompose(
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
    while current_rank < max_rank
        cpd = als_optimize(A, cpd; alg, check, maxiter, verbose);
        if 1.0 - ITensorCPD.fit(check) < epsilon
            return cpd
        else
            current_rank += rank_step;
            println("\nIncreasing rank to: $(current_rank)")    
        end
        cpd = increase_cpd_rank(cpd, Index(current_rank, tags(cp_rank(cpd))); rng)
    end

end

function increase_cpd_rank(cpd::ITensorCPD.CPD, new_rank::Index; rng = nothing)
    rng = isnothing(rng) ? MersenneTwister(3) : rng
    elt = eltype(cpd)

    old_rank = cp_rank(cpd)
    @assert dim(new_rank) ≥ dim(old_rank)
    
    updated_factors = Vector{ITensor}()
    for i in 1:length(cpd.factors)
        is = ind(cpd[i], 2)
        it = random_itensor(rng, elt, new_rank, is)
        rtensor, l = row_norm(it, is)

        array(rtensor)[1:dim(old_rank),:] = array(cpd[i])[:,:]
        push!(updated_factors, rtensor)
    end
    lambda = itensor(similar(data(cpd.λ), dim(new_rank)), new_rank)
    return ITensorCPD.CPD{paramT(cpd)}(updated_factors, lambda)
end