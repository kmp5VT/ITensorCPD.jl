using ITensors.NDTensors.Expose

function ldiv_solve!!(A::Exposed, B::Exposed; factorizeA = false)
    return ldiv_solve!(unexpose(A), unexpose(B); factorizeA)
end

function ldiv_solve!(A, B; factorizeA = false)
    A = factorizeA ? factorize(A) : A
    return A \ B
end