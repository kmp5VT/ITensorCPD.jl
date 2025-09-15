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
        real(sqrt(abs(check.ref_norm * check.ref_norm + fact_square - 2 * abs(inner_prod))))
    curr_fit = 1.0 - real(normResidual / check.ref_norm)
    Δfit = abs(check.lastfit - curr_fit)
    check.lastfit = (curr_fit)

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

function norm_factors(partial_gram::Vector, λ::ITensor)
    had = copy(partial_gram[1])
    for i = 2:length(partial_gram)
        hadamard_product!(had, had, partial_gram[i])
    end
    return (had*(λ*dag(prime(λ))))[]
end

CPDFit(check::FitCheck) = check.final_fit