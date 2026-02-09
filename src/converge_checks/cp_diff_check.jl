## In this code I will compute the norm difference between two CP 
## approximations from consequtive iterations. If the distance is small the 
## calculation is stopped. 
## || \hat{T}_{n} - \hat{T}_{n-1} || / || \hat{T}_{n-1} || < ϵ 

mutable struct CPDiffCheck <: ConvergeAlg
    iter::Int
    counter::Int
    tolerance::Number
    max_counter::Int
    norm_prev_iter::Number
    PrevCP
    lastfit::Number
    final_fit::Number
    total_iter::Int

    CPDiffCheck(tol, max) = new(0, 0, tol, max, 0.0, nothing, 1, 0, 0)
end

function check_converge(check::CPDiffCheck, factors, λ, partial_gram; verbose = true)
    check.iter += 1
    rank = ind(λ, 1)

    if isnothing(check.PrevCP)
        check.PrevCP = dag(CPD{ITensor}(prime.(factors; tags=tags(rank)), prime(λ)))
        check.norm_prev_iter  = norm_factors([i * prime(dag(i); tags=tags(rank)) for i in factors], λ)
        return false
    end

    currCP = CPD{ITensor}(factors, λ)
    inner_prod = real((check.PrevCP.λ * cp_cp_contract(check.PrevCP, currCP)[1] * λ)[])
    fact_square = norm_factors([i * prime(dag(i); tags=tags(rank)) for i in factors], λ)
    normResidual =
        sqrt(abs(check.norm_prev_iter + fact_square - 2 * abs(inner_prod)))
    curr_fit = one(eltype(inner_prod)) - (normResidual / sqrt(check.norm_prev_iter))
    Δfit = abs(check.lastfit - curr_fit)
    check.lastfit = curr_fit

    check.PrevCP = dag(CPD{ITensor}(prime.(factors; tags=tags(rank)), prime(λ)))
    check.norm_prev_iter  = fact_square

    if (verbose)
        println("$(dim(rank))\t $(check.iter) \t $(curr_fit) \t $(Δfit)")
    end

    if Δfit < check.tolerance
        check.counter += 1
        if check.counter >= 2
            check.total_iter = check.iter
            check.iter = 0
            check.counter = 0
            check.final_fit = check.lastfit
            check.lastfit = 0
            check.PrevCP = nothing
            return true
        end
    else
        check.counter = 0
    end

    if check.iter == check.max_counter
        check.total_iter = check.iter
        check.iter = 0
        check.counter = 0
        check.final_fit = check.lastfit
        check.lastfit = 0
        check.PrevCP = nothing
    end

    return false
end

CPDFit(check::CPDiffCheck) = check.final_fit