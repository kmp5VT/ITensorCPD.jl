## In this code I will compute angle between two CPD
## approximations from consequtive iterations. If the change in the 
## angle is small the calculation is stopped. 
## acos(\hat{T}_{n} . \hat{T}_{n-1} / (|| \hat{T}_{n} || * || \hat{T}_{n-1} ||) 

mutable struct CPAngleCheck <: ConvergeAlg
    iter::Int
    counter::Int
    tolerance::Number
    max_counter::Int
    norm_prev_iter::Number
    PrevCP
    lastangle::Number
    final_fit::Number

    CPAngleCheck(tol, max) = new(0, 0, tol, max, 0.0, nothing, 1, 0)
end

function check_converge(check::CPAngleCheck, factors, λ, partial_gram; verbose = true)
    check.iter += 1
    rank = ind(λ, 1)

    if isnothing(check.PrevCP)
        check.PrevCP = dag(CPD{ITensor}(prime.(factors; tags=tags(rank)), prime(λ)))
        check.norm_prev_iter  = sqrt(norm_factors([i * prime(dag(i); tags=tags(rank)) for i in factors], λ))
        return false
    end

    currCP = CPD{ITensor}(factors, λ)
    numer = real((check.PrevCP.λ * cp_cp_contract(check.PrevCP, currCP)[1] * λ)[])
    norm_currCP = sqrt(norm_factors([i * prime(dag(i); tags=tags(rank)) for i in factors], λ))
    
    theta = numer / (norm_currCP * check.norm_prev_iter)
    elt = real(eltype(currCP))
    theta = theta > one(elt) ? one(elt) : theta
    curr_angle = acos(theta)
    Δfit = abs(check.lastangle - curr_angle)
    check.lastangle = curr_angle

    check.PrevCP = dag(CPD{ITensor}(prime.(factors; tags=tags(rank)), prime(λ)))
    check.norm_prev_iter  = norm_currCP

    if (verbose)
        println("$(dim(rank))\t $(check.iter) \t $(curr_angle) \t $(Δfit)")
    end

    if Δfit < check.tolerance
        check.counter += 1
        if check.counter >= 2
            check.iter = 0
            check.counter = 0
            check.final_fit = check.lastangle
            check.lastangle = 0
            check.PrevCP = nothing
            return true
        end
    else
        check.counter = 0
    end

    if check.iter == check.max_counter
        check.iter = 0
        check.counter = 0
        check.final_fit = check.lastangle
        check.lastangle = 0
        check.PrevCP = nothing
    end

    return false
end

CPDFit(check::CPAngleCheck) = check.final_fit