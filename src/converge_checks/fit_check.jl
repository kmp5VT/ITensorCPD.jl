## In this code I will compute the Fit (norm difference) between a CP and the true tensor.
##  If the distance between two consecutive fits are small, the calculation is stopped.
## || T - \hat{T}_{n} || / || T || < ϵ 

mutable struct FitCheck <: ConvergeAlg
    iter::Int
    counter::Int
    tolerance::Number
    max_counter::Int
    ref_norm::Number
    MttKRP::ITensor
    lastfit::Number
    final_fit::Number

    FitCheck(tol, max, norm) = new(0, 0, tol, max, norm, ITensor(), 1, 0)
end

save_mttkrp(check::FitCheck, mttkrp::ITensor) = check.MttKRP = mttkrp

function check_converge(check::FitCheck, factors, λ, partial_gram; verbose = true)
    check.iter += 1
    rank = ind(partial_gram[1], 1)
    inner_prod = 0
    inner_prod = sum(hadamard_product(check.MttKRP, had_contract(dag(factors[end]), dag(λ), rank)))
    fact_square = norm_factors(partial_gram, λ)
    normResidual =
        sqrt(abs(check.ref_norm * check.ref_norm + fact_square - 2 * abs(inner_prod)))
    curr_fit = 1.0 - (normResidual / check.ref_norm)
    Δfit = abs(check.lastfit - curr_fit)
    check.lastfit = curr_fit

    if (verbose)
        println("$(dim(rank))\t $(check.iter) \t $(curr_fit) \t $(Δfit)")
    end

    if Δfit < check.tolerance
        check.counter += 1
        if check.counter >= 2
            check.iter = 0
            check.counter = 0
            check.final_fit = check.lastfit
            check.lastfit = 0
            return true
        end
    else
        check.counter = 0
    end

    if check.iter == check.max_counter
        check.iter = 0
        check.counter = 0
        check.final_fit = check.lastfit
        check.lastfit = 0
    end

    return false
end

CPDFit(check::FitCheck) = check.final_fit