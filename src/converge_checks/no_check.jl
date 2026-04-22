mutable struct NoCheck <: ConvergeAlg
    iter::Int
    max_counter::Int
    lastfit::Number

    NoCheck(max) = new(0, max, -1)
end

function check_converge(check::NoCheck, factors, λ, partial_gram; verbose = false)
    check.iter += 1
    rank = ind(λ, 1)
    if (verbose)
        println("$(dim(rank))\t $(check.iter)")
    end
    if check.iter == check.max_counter
        return true
    end
    return false
end
