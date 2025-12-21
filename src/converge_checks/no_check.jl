mutable struct NoCheck <: ConvergeAlg
    counter::Int
    max_counter::Int
    lastfit::Number

    NoCheck(max) = new(0, max, -1)
end

function check_converge(check::NoCheck, factors, λ, partial_gram; verbose = false)
    rank = ind(λ, 1)
    if (verbose)
        println("$(dim(rank))\t $(check.counter)")
    end
    if check.counter ≥ check.max_counter
        return true
    end
    check.counter += 1
    return false
end
