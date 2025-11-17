using ITensors.NDTensors.Expose
using GPUArraysCore: AbstractGPUArray

function ldiv_solve!!(A::Exposed, B::Exposed; factorizeA = false)
    return ldiv_solve!(unexpose(A), unexpose(B); factorizeA)
end

function ldiv_solve!!(A::Exposed{<:AbstractGPUArray}, B::Exposed{<:AbstractGPUArray}; factorizeA = false)
    return ldiv_solve!(A,B; factorizeA = false)
end

## For now just call with QR CP if factorize. Later this will be more complex.
function ldiv_solve!(A, B; factorizeA = false)
    if factorizeA 
        return qr(A, ColumnNorm()) \ B
    else
        return A \ B
    end
end