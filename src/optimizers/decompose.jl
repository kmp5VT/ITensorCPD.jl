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
